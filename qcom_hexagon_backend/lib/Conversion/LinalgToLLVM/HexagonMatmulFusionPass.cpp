//===- HexagonMatmulFusionPass.cpp - matmul fusion   -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass uses linalg-fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"

#define DEBUG_TYPE "hexagon-matmul-fusion"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hexagon {

#define GEN_PASS_DEF_HEXAGONMATMULFUSION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

// Tile size for the fused contraction (reduction) dimension of the fc2 matmul.
// The fc1 result is produced one K-slice at a time and consumed immediately,
// so the [M x K_ff] activation is never fully materialized.
static constexpr int64_t kReductionTileSize = 128;

// True if `op` has at least one reduction iterator and can be tiled.
static bool isTileableReduction(linalg::GenericOp op) {
  if (!isa<TilingInterface>(op.getOperation()))
    return false;
  return llvm::is_contained(op.getIteratorTypesArray(),
                            utils::IteratorType::reduction);
}

// True if the body of `op` computes a tanh-approximation GELU. This is the
// fingerprint that distinguishes the MLP down-projection (fc2, which folds the
// activation into its epilogue) from a plain matmul.
static bool hasTanhGeluBody(linalg::GenericOp op) {
  return op.getBody()
      ->walk([](Operation *inner) {
        return isa<math::TanhOp>(inner) ? WalkResult::interrupt()
                                        : WalkResult::advance();
      })
      .wasInterrupted();
}

// True if `operand` of `op` is indexed by one of `op`'s reduction dimensions,
// i.e. the value flowing through it is contracted - it feeds the matmul's K
// loop rather than being a plain elementwise input.
static bool operandFeedsContraction(linalg::GenericOp op, OpOperand *operand) {
  SmallVector<utils::IteratorType> iterators = op.getIteratorTypesArray();
  AffineMap map =
      cast<linalg::LinalgOp>(op.getOperation()).getMatchingIndexingMap(operand);
  for (AffineExpr result : map.getResults())
    if (auto dim = dyn_cast<AffineDimExpr>(result))
      if (iterators[dim.getPosition()] == utils::IteratorType::reduction)
        return true;
  return false;
}

// If `consumer` is an fc2-like op (tanh-GELU epilogue + a single contraction),
// return the fc1 matmul feeding its contraction input; otherwise nullptr.
static linalg::GenericOp findFusableFc1(linalg::GenericOp consumer,
                                       bool allowRecompute) {
  if (!isTileableReduction(consumer) || !hasTanhGeluBody(consumer))
    return nullptr;
  for (OpOperand *input : consumer.getDpsInputOperands()) {
    auto producer = input->get().getDefiningOp<linalg::GenericOp>();
    if (!producer || !isTileableReduction(producer))
      continue;
    if (!operandFeedsContraction(consumer, input))
      continue;
    if (!producer->hasOneUse() && !allowRecompute)
      continue;
    return producer;
  }
  return nullptr;
}

// Tile `consumer`'s reduction dimension(s) and fuse `producer` (fc1) into the
// resulting loop, so each fc1 activation tile is GELU'd and contracted in
// place instead of the full [M x K_ff] tensor being written out.
static LogicalResult fuseFc1IntoFc2(RewriterBase &rewriter,
                                    linalg::GenericOp producer,
                                    linalg::GenericOp consumer) {
  // Give the consumer a private zero-init before tiling, so the fused scf.for
  // carries its own accumulator rather than an `iter_args` that aliases a
  // `linalg.fill` shared with other ops (both MLP layers reuse one fill; the
  // downstream bufferizer would otherwise let the second loop see the first
  // loop's result as its initial value).
  {
    OpOperand *initOperand = consumer.getDpsInitOperand(0);
    if (auto fill = initOperand->get().getDefiningOp<linalg::FillOp>()) {
      if (!fill->hasOneUse()) {
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(consumer);
        auto ty = cast<RankedTensorType>(initOperand->get().getType());
        Value empty = tensor::EmptyOp::create(rewriter, consumer.getLoc(),
                                              ty.getShape(), ty.getElementType());
        Value privateFill =
            linalg::FillOp::create(rewriter, consumer.getLoc(),
                                   fill.getInputs(), ValueRange{empty})
                .getResult(0);
        rewriter.modifyOpInPlace(consumer,
                                 [&] { initOperand->set(privateFill); });
      }
    }
  }

  SmallVector<OpFoldResult> tileSizes;
  for (utils::IteratorType it : consumer.getIteratorTypesArray())
    tileSizes.push_back(rewriter.getIndexAttr(
        it == utils::IteratorType::reduction ? kReductionTileSize : 0));

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);
  // FullReduction (the default): a plain scf.for over the contraction that
  // threads the fc2 accumulator through iter_args - not split-k.

  Operation *producerOp = producer.getOperation();
  scf::SCFTileAndFuseOptions options;
  options.setTilingOptions(tilingOptions);
  options.setFusionControlFn(
      [producerOp](tensor::ExtractSliceOp, OpResult originalProducer, bool)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (originalProducer.getOwner() != producerOp)
          return std::nullopt;
        return scf::SCFTileAndFuseOptions::ControlFnResult{
            /*yieldProducerReplacement=*/false};
      });

  rewriter.setInsertionPoint(consumer);
  FailureOr<scf::SCFTileAndFuseResult> result =
      scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, cast<TilingInterface>(consumer.getOperation()), options);
  if (failed(result))
    return failure();

  // tileConsumerAndFuseProducersUsingSCF does not replace the original
  // consumer; splice in its loop-carried result and drop the dead fc1.
  auto repl = result->replacements.find(consumer.getResult(0));
  if (repl == result->replacements.end())
    return failure();
  rewriter.replaceOp(consumer, repl->second);
  if (producer->use_empty())
    rewriter.eraseOp(producer);
  return success();
}

struct HexagonMatmulFusionPass
    : public impl::HexagonMatmulFusionBase<HexagonMatmulFusionPass> {
  HexagonMatmulFusionPass() = default;
  explicit HexagonMatmulFusionPass(const HexagonMatmulFusionOptions &options)
      : HexagonMatmulFusionBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect, math::MathDialect,
                    arith::ArithDialect>();
    // linalg.generic / tensor slice ops need their TilingInterface external
    // models for scf tile-and-fuse to work when this pass runs outside a tool
    // that already calls registerAllExtensions().
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
  }

  static bool isMatmulOp(linalg::GenericOp genericOp) {
    auto linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
    if (!linalg::isaContractionOpInterface(linalgOp))
      return false;
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
      return false;
    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(linalgOp);
    return succeeded(dims) && dims->batch.empty() && dims->m.size() == 1 &&
           dims->n.size() == 1 && dims->k.size() == 1;
  }

#define tileM 64
#define tileK 64
#define tileN 64
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    IRRewriter rewriter(context);

    llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    llvm::outs() << "Running Matmul Fusion Pass\n";
    llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    llvm::outs() << "tileM: "<<tileM<<" tileK: "<<tileK<<" tileN: "<<tileN<<"\n";
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(false, false);
    llvm::outs() << "\n=== Before fusion ===\n";
    funcOp.print(llvm::outs(), flags);
    llvm::outs() << "\n";
    llvm::SmallVector<mlir::linalg::GenericOp> matmuls;
    funcOp.walk([&](mlir::linalg::GenericOp op) {
      if (isMatmulOp(op)) {
        matmuls.push_back(op);
      }
    });
    // Identify the producer (A * B = C) and the consumer (C * D = E)
    if (matmuls.size() >= 2) {
      auto producerMatmul = matmuls[0]; // %4 in your IR
      auto consumerMatmul = matmuls[1]; // %7 in your IR
      for (mlir::OpOperand &operand : consumerMatmul->getOpOperands()) {
        if (auto definingGeneric = operand.get().getDefiningOp<mlir::linalg::GenericOp>()) {
          if (!isMatmulOp(definingGeneric)) {
            for (mlir::Value addInput : definingGeneric.getDpsInputs()) {
              if (addInput == producerMatmul.getResult(0)) {
                rewriter.modifyOpInPlace(consumerMatmul, [&]() {
                  consumerMatmul->setOperand(operand.getOperandNumber(), producerMatmul.getResult(0));
                });
                llvm::outs()<<"Found two matmuls and replace 2's input with 1's output.\n";
                break;
              }
            }
          }
        }
      }
    } else {
      llvm::errs() << "Cannot find producer.\n";
      return signalPassFailure();
    }

    linalg::GenericOp producer = matmuls[0];
    linalg::GenericOp consumer = matmuls[1];
    // ------------------------------------------------------------
    // Step 1: Tile parallel dimensions (d0, d2) using scf.forall AND FUSE
    // ------------------------------------------------------------
    scf::SCFTilingOptions forallTilingOptions;
    SmallVector<OpFoldResult> forallTileSizes = {
        rewriter.getIndexAttr(tileM),
        rewriter.getIndexAttr(0),       // d1 is reduction, skip for now
        rewriter.getIndexAttr(tileN),
    };
    forallTilingOptions.setTileSizes(forallTileSizes);
    //forallTilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

    scf::SCFTileAndFuseOptions forallTileAndFuseOptions;
    forallTileAndFuseOptions.setTilingOptions(forallTilingOptions);
    auto fusionControlFn1 = [&](tensor::ExtractSliceOp extractSlice,
        OpResult producerResult, bool isDestinationOperand) ->
            std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      if (producerResult.getOwner() == producer.getOperation()) {
        scf::SCFTileAndFuseOptions::ControlFnResult result;
        result.yieldProducerReplacement = false;
        return result;
      }
      return std::nullopt;
    };
    forallTileAndFuseOptions.setFusionControlFn(fusionControlFn1);
    rewriter.setInsertionPoint(consumer);
    FailureOr<scf::SCFTileAndFuseResult> forallResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(consumer.getOperation()), forallTileAndFuseOptions);
    if (failed(forallResult)) {
      funcOp.emitWarning("Matmul fusion: step 1 (forall) failed.");
      return signalPassFailure();
    }
    Value consumerReplacement1;
    for (auto &[oldValue, newValue] : forallResult->replacements) {
      if (oldValue == consumer->getResult(0)) {
        consumerReplacement1 = newValue;
        break;
      }
    }
    if (consumerReplacement1) {
      rewriter.replaceOp(consumer, consumerReplacement1);
    }

    linalg::GenericOp tiledConsumer;
    linalg::GenericOp tiledProducer;
    Operation *forallLoop = forallResult->loops.front();
    forallLoop->walk([&](linalg::GenericOp op) {
      for (auto operand : op->getOperands()) {
        if (auto def = operand.getDefiningOp<linalg::GenericOp>()) {
          tiledProducer = def;
          tiledConsumer = op;
        }
      }
    });
    if (!tiledConsumer || !tiledProducer) {
      llvm::errs() << "Could not find tiled consumer or producer after step 1.\n";
      return signalPassFailure();
    }
    llvm::outs() << "\n=== After Step 1 ===\n";
    funcOp.print(llvm::outs(), flags);
    llvm::outs() << "\n=== Verifying after fusion ===\n";
    if (failed(mlir::verify(funcOp))) {
      llvm::errs() << "!!! VERIFICATION FAILED !!!\n";
      return signalPassFailure();
    }
    llvm::outs() << "!!! VERIFICATION PASSED !!!\n";
    // ------------------------------------------------------------
    // Step 2: Tile reduction dimension (d1) using scf.for AND FUSE
    // ------------------------------------------------------------
    scf::SCFTilingOptions forTilingOptions;
    SmallVector<OpFoldResult> forTileSizes = {
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(tileK), // d1 is reduction
        rewriter.getIndexAttr(0),
    };
    forTilingOptions.setTileSizes(forTileSizes);

    scf::SCFTileAndFuseOptions forTileAndFuseOptions;
    forTileAndFuseOptions.setTilingOptions(forTilingOptions);
    auto fusionControlFn2 = [&](tensor::ExtractSliceOp extractSlice,
        OpResult producerResult, bool isDestinationOperand) ->
            std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      // Look for the producer that was generated by Step 1
      if (producerResult.getOwner() == tiledProducer.getOperation()) {
        scf::SCFTileAndFuseOptions::ControlFnResult result;
        result.yieldProducerReplacement = false;
        return result;
      }
      return std::nullopt;
    };
    forTileAndFuseOptions.setFusionControlFn(fusionControlFn2);
    rewriter.setInsertionPoint(tiledConsumer);
    FailureOr<scf::SCFTileAndFuseResult> fuseResult = scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, cast<TilingInterface>(tiledConsumer.getOperation()), forTileAndFuseOptions);
    if (failed(fuseResult)) {
      funcOp.emitWarning("Matmul fusion: step 2 failed.");
      return signalPassFailure();
    } else {
      llvm::outs()<<"Successfully tiled the consumer and fused producer.\n";
    }
    if (fuseResult->loops.empty()) {
      llvm::outs()<<"Matmul fusion: step 2 produced no loops.\n";
      return signalPassFailure();
    }
    Operation *loop = fuseResult->loops.front();
    // ---------------------------------------------------------
    // Localize the Producer's Accumulator
    linalg::GenericOp fusedProducerInsideLoop;
    loop->walk([&](linalg::GenericOp op) {
        for (Operation *user : op->getUsers()) {
            if (isa<linalg::GenericOp>(user)) {
                fusedProducerInsideLoop = op;
            }
        }
    });
    if (fusedProducerInsideLoop) {
        Value outs = fusedProducerInsideLoop.getDpsInitOperand(0)->get();
        if (auto extractOp = outs.getDefiningOp<tensor::ExtractSliceOp>()) {
            rewriter.setInsertionPoint(fusedProducerInsideLoop);
            auto sliceType = cast<RankedTensorType>(extractOp.getType());
            Type elemType = sliceType.getElementType();
            auto localEmpty = tensor::EmptyOp::create(
                rewriter,
                fusedProducerInsideLoop.getLoc(),
                sliceType.getShape(),
                sliceType.getElementType()
            );
            auto cst0 = arith::ConstantOp::create(
                rewriter,
                fusedProducerInsideLoop.getLoc(),
                rewriter.getZeroAttr(elemType)
            );
            auto localFill = linalg::FillOp::create(
                rewriter,
                fusedProducerInsideLoop.getLoc(),
                ValueRange{cst0.getResult()},
                ValueRange{localEmpty.getResult()}
            );
            rewriter.modifyOpInPlace(fusedProducerInsideLoop, [&]() {
                fusedProducerInsideLoop.getDpsInitOperand(0)->set(localFill.getResult(0));
            });
        }
    }
    // ------------------------------------------------------------
    // Localize the Consumer's Accumulator
    if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
      OpOperand &initOperand = forOp.getInitArgsMutable()[0];
      Value initVal = initOperand.get();
      if (auto extractOp = initVal.getDefiningOp<tensor::ExtractSliceOp>()) {
        rewriter.setInsertionPoint(forOp);
        auto sliceType = cast<RankedTensorType>(extractOp.getType());
        Type elemType = sliceType.getElementType();
        auto localEmpty = tensor::EmptyOp::create(
            rewriter, forOp.getLoc(),
            sliceType.getShape(), sliceType.getElementType());
        auto cst0 = arith::ConstantOp::create(
            rewriter, forOp.getLoc(),
            rewriter.getZeroAttr(elemType)
        );
        auto localFill = linalg::FillOp::create(
            rewriter, forOp.getLoc(),
            ValueRange{cst0.getResult()}, ValueRange{localEmpty.getResult()});
        rewriter.modifyOpInPlace(forOp, [&]() {
          initOperand.set(localFill.getResult(0));
        });
      }
    }
    // ------------------------------------------------------------
    Value consumerReplacement;
    for (auto &[oldValue, newValue] : fuseResult->replacements) {
      if (oldValue == tiledConsumer->getResult(0)) {
        consumerReplacement = newValue;
        break;
      }
    }
    if (!consumerReplacement) {
      llvm::outs()<<"missing consumer replacement\n";
    } else {
      llvm::outs()<<"consumer has replacement\n";
      tiledConsumer->getResult(0).replaceUsesWithIf(
        consumerReplacement,
        [&](OpOperand &use) {
          return !loop->isProperAncestor(use.getOwner());
      });
    }
    llvm::outs() << "\n=== After Step 2 ===\n";
    funcOp.print(llvm::outs(), flags);
    llvm::outs() << "\n";
    // ------------------------------------------------------------
    // Clean up the Producer's materialization to allow dead code elimination
    llvm::SmallVector<Operation*> opsToErase;
    for (auto *user : producer->getResult(0).getUsers()) {
      if (isa<bufferization::MaterializeInDestinationOp>(user)) {
        opsToErase.push_back(user);
      }
      else if (auto genericUser = dyn_cast<linalg::GenericOp>(user)) {
        if (!isMatmulOp(genericUser)) { // It is the bypassed Add op
          for (auto *addUser : genericUser->getResult(0).getUsers()) {
            if (isa<bufferization::MaterializeInDestinationOp>(addUser)) {
              opsToErase.push_back(addUser);
            }
          }
        }
      }
    }
    for (auto *op : opsToErase) {
      rewriter.eraseOp(op);
    }
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    bufferization::populateEmptyTensorToAllocTensorPattern(patterns);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    llvm::outs() << "\n=== After Step 3 ===\n";
    funcOp.print(llvm::outs(), flags);
    llvm::outs() << "\n";
    // ------------------------------------------------------------
    OpPassManager pm;
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(createLoopInvariantCodeMotionPass());
      pm.addPass(createSCCPPass());
    (void)runPipeline(pm, funcOp);
    llvm::outs() << "\n=== After fusion ===\n";
    funcOp.print(llvm::outs(), flags);
    llvm::outs() << "\n";
    /*FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Dump the function around fusion when MLIR_ENABLE_DUMP=1 (repo convention)
    // or `-debug-only=hexagon-matmul-fusion` is set.
    bool dump = isEnvTrue("MLIR_ENABLE_DUMP");
    auto dumpFunc = [&](const Twine &stage) {
      if (dump) {
        llvm::outs() << "\n=== hexagon-matmul-fusion: " << stage << " ===\n";
        funcOp.print(llvm::outs());
        llvm::outs() << "\n";
      }
      LLVM_DEBUG({
        DBGS() << "=== " << stage << " ===\n";
        funcOp.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
    };
    dumpFunc("before");

    // Collect (fc1, fc2) pairs up front; fusing mutates the IR.
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> pairs;
    funcOp.walk([&](linalg::GenericOp consumer) {
      if (linalg::GenericOp producer =
              findFusableFc1(consumer, fusionAllowRecompute))
        pairs.emplace_back(producer, consumer);
    });

    IRRewriter rewriter(context);
    unsigned fused = 0;
    for (auto &[producer, consumer] : pairs) {
      if (!producer->getBlock() || !consumer->getBlock())
        continue; // invalidated by an earlier fusion
      if (succeeded(fuseFc1IntoFc2(rewriter, producer, consumer)))
        ++fused;
      else
        consumer->emitWarning(
            "hexagon-matmul-fusion: fc1->GELU->fc2 fusion failed");
    }

    if (fused == 0) {
      LLVM_DEBUG(DBGS() << "no fc1->GELU->fc2 chains fused\n");
      dumpFunc("after (0 fused)");
      return;
    }

    // Clean up the slices and now-dead ops left by tile-and-fuse.
    RewritePatternSet patterns(context);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    // Turn extract_slice(fill) of the fused fc1 accumulator into a loop-local
    // fill so the full [M x K_ff] zero buffer is not left behind.
    linalg::populateSwapExtractSliceWithFillPatterns(patterns);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      funcOp->emitWarning("hexagon-matmul-fusion: cleanup did not converge");

    dumpFunc("after (" + Twine(fused) + " fused)");*/
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonMatmulFusionPass(const HexagonMatmulFusionOptions &options) {
  return std::make_unique<HexagonMatmulFusionPass>(options);
}

} // namespace hexagon
} // namespace mlir
