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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
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

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16

namespace mlir {
namespace hexagon {

#define GEN_PASS_DEF_HEXAGONMATMULFUSION
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct HexagonMatmulFusionPass : public impl::HexagonMatmulFusionBase<HexagonMatmulFusionPass> {
  HexagonMatmulFusionPass() = default;
  explicit HexagonMatmulFusionPass(const HexagonMatmulFusionOptions &options)
      : HexagonMatmulFusionBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    IRRewriter rewriter(context);
    llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    llvm::outs() << "Running Hexagon Matmul Fusion Pass\n";
    llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    llvm::outs() << "\n=== Before fusion ===\n";
    funcOp.print(llvm::outs());
    llvm::outs() << "\n";
    // Identify the producer (A * B = C) and the consumer (C * D = E)
    linalg::GenericOp consumer;
    linalg::GenericOp producer;
    funcOp.walk([&](linalg::GenericOp op) {
      for (auto operand : op->getOperands()) {
        if (auto def = operand.getDefiningOp<linalg::GenericOp>()) {
          producer = def;
          consumer = op;
        }
      }
    });
    if (!producer) {
      llvm::errs() << "Cannot find producer.\n";
      return signalPassFailure();
    }
    /*if (producer.getDpsInitOperand(0)->get() == consumer.getDpsInitOperand(0)->get()) {
      Value sharedAcc = producer.getDpsInitOperand(0)->get();
      if (auto fillOp = sharedAcc.getDefiningOp<linalg::FillOp>()) {
        rewriter.setInsertionPoint(producer);
        IRMapping mapping;
        Value fillOuts = fillOp.getDpsInitOperand(0)->get();
        if (Operation *emptyOp = fillOuts.getDefiningOp()) {
          Operation *clonedEmpty = rewriter.clone(*emptyOp, mapping);
          mapping.map(fillOuts, clonedEmpty->getResult(0));
        }
        auto clonedFill = cast<linalg::FillOp>(rewriter.clone(*fillOp.getOperation(), mapping));
        rewriter.modifyOpInPlace(producer, [&]() {
          producer.getDpsInitOperand(0)->set(clonedFill.getResult(0));
        });
      }
    }*/
    // ------------------------------------------------------------
    scf::SCFTilingOptions tilingOptions;
    SmallVector<OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(BLOCK_M),
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0)
    };
    tilingOptions.setTileSizes(tileSizes);
    //tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);
    auto fusionControlFn = [&](tensor::ExtractSliceOp extractSlice, 
        OpResult producerResult, bool isDestinationOperand) ->
            std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      if (producerResult.getOwner() == producer.getOperation()) {
        scf::SCFTileAndFuseOptions::ControlFnResult result;
        result.yieldProducerReplacement = false;
        return result;
      }
      return std::nullopt;
    };
    tileAndFuseOptions.setFusionControlFn(fusionControlFn);
    rewriter.setInsertionPoint(consumer);
    auto fuseResult = scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, cast<TilingInterface>(consumer.getOperation()), tileAndFuseOptions);
    if (failed(fuseResult)) {
      funcOp.emitWarning("Hexagon Matmul fusion failed.");
      return signalPassFailure();
    } else {
      llvm::outs()<<"tile the consumer and fuse producer.\n";
    }
    Operation *loop = fuseResult->loops.front();
    // Localize the Producer's Accumulator
    // ---------------------------------------------------------
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
            auto localEmpty = tensor::EmptyOp::create(
                rewriter,
                fusedProducerInsideLoop.getLoc(),
                sliceType.getShape(),
                sliceType.getElementType()
            );
            auto cst0 = arith::ConstantOp::create(
                rewriter,
                fusedProducerInsideLoop.getLoc(),
                rewriter.getF16FloatAttr(0.0)
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
    // ---------------------------------------------------------
    for (auto &[oldV, newV] : fuseResult->replacements) {
      llvm::errs() << "Old value users:\n";
      for (Operation *user : oldV.getUsers()) {
        llvm::errs() << "  ";
        user->print(llvm::errs());
        llvm::errs() << "\n";
        llvm::errs() << "  inside loop = "
                 << loop->isProperAncestor(user)
                 << "\n";
      }
    }
    Value consumerReplacement;
    for (auto &[oldValue, newValue] : fuseResult->replacements) {
      if (oldValue == consumer->getResult(0)) {
        consumerReplacement = newValue;
        break;
      }
    }
    if (!consumerReplacement) {
      llvm::outs()<<"missing consumer replacement\n";
    } else {
      llvm::outs()<<"consumer has replacement\n";
      consumer->getResult(0).replaceUsesWithIf(
        consumerReplacement,
        [&](OpOperand &use) {
          return !loop->isProperAncestor(use.getOwner());
      });
    }
    for (auto *user : llvm::make_early_inc_range(producer->getResult(0).getUsers())) {
      if (isa<bufferization::MaterializeInDestinationOp>(user)) {
        rewriter.eraseOp(user);
      }
    }
    llvm::errs() << "Replacement count = "
             << fuseResult->replacements.size() << "\n";
    for (auto &[oldV, newV] : fuseResult->replacements) {
      llvm::errs() << "=== Replacement ===\n";
      llvm::errs() << "old value = ";
      oldV.print(llvm::errs());
      llvm::errs() << "\n";
      llvm::errs() << "new value = ";
      newV.print(llvm::errs());
      llvm::errs() << "\n";
    }
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    bufferization::populateEmptyTensorToAllocTensorPattern(patterns);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    OpPassManager pm;
      //pm.addPass(bufferization::createPromoteAllocsToStackPass());
      //pm.addPass(createCanonicalizerPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(createLoopInvariantCodeMotionPass());
      pm.addPass(createSCCPPass());
    (void)runPipeline(pm, funcOp);
    /*if (consumer->use_empty()) rewriter.eraseOp(consumer);
    for (Operation *user : llvm::make_early_inc_range(producer->getUsers())) {
      if (isa<tensor::ExtractSliceOp>(user) && user->use_empty()) {
        rewriter.eraseOp(user);
      }
    }
    if (producer->use_empty()) rewriter.eraseOp(producer);*/
    llvm::outs() << "\n=== After fusion ===\n";
    funcOp.print(llvm::outs());
    llvm::outs() << "\n";
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createHexagonMatmulFusionPass(const HexagonMatmulFusionOptions &options) {
  return std::make_unique<HexagonMatmulFusionPass>(options);
}

} // namespace hexagon
} // namespace mlir
