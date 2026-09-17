// RUN: linalg-hexagon-opt %s --pass-pipeline="builtin.module(func.func(hexagon-matmul-fusion))" -split-input-file | FileCheck %s

// fc1 (matmul) -> tanh-approx GELU -> fc2 (matmul) should be fused: the fc1
// [M x K_ff] activation is produced a K-slice at a time inside the fc2
// contraction loop instead of being written out in full.

#mmA = affine_map<(m, k, n) -> (m, k)>
#mmB = affine_map<(m, k, n) -> (k, n)>
#mmC = affine_map<(m, k, n) -> (m, n)>
#bias = affine_map<(m, k, n) -> (k)>

// CHECK-LABEL: func.func @mlp_fc1_gelu_fc2
// CHECK-SAME:    %[[X:.*]]: tensor<8x16xf32>, %[[W1:.*]]: tensor<16x256xf32>, %[[B1:.*]]: tensor<256xf32>, %[[W2:.*]]: tensor<256x8xf32>
func.func @mlp_fc1_gelu_fc2(%x: tensor<8x16xf32>, %w1: tensor<16x256xf32>,
                            %b1: tensor<256xf32>, %w2: tensor<256x8xf32>)
    -> tensor<8x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %half = arith.constant 5.000000e-01 : f32
  %one = arith.constant 1.000000e+00 : f32
  %c = arith.constant 7.978845e-01 : f32

  %e1 = tensor.empty() : tensor<8x256xf32>
  %f1 = linalg.fill ins(%cst : f32) outs(%e1 : tensor<8x256xf32>) -> tensor<8x256xf32>

  // fc1: [8x16] x [16x256] -> [8x256]
  %fc1 = linalg.generic {
      indexing_maps = [#mmA, #mmB, #mmC],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%x, %w1 : tensor<8x16xf32>, tensor<16x256xf32>)
      outs(%f1 : tensor<8x256xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %o, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x256xf32>

  %e2 = tensor.empty() : tensor<8x8xf32>
  %f2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<8x8xf32>) -> tensor<8x8xf32>

  // fc2 with tanh-approx GELU folded onto the fc1 activation, contraction K=256.
  %fc2 = linalg.generic {
      indexing_maps = [#mmA, #bias, #mmB, #mmC],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%fc1, %b1, %w2 : tensor<8x256xf32>, tensor<256xf32>, tensor<256x8xf32>)
      outs(%f2 : tensor<8x8xf32>) {
  ^bb0(%h: f32, %bb: f32, %w: f32, %o: f32):
    %hb = arith.addf %h, %bb : f32
    %hb3 = arith.mulf %hb, %hb : f32
    %hb3b = arith.mulf %hb3, %hb : f32
    %inner = arith.mulf %hb3b, %c : f32
    %inner2 = arith.addf %hb, %inner : f32
    %t = math.tanh %inner2 : f32
    %t1 = arith.addf %t, %one : f32
    %g0 = arith.mulf %hb, %half : f32
    %gelu = arith.mulf %g0, %t1 : f32
    %mm = arith.mulf %gelu, %w : f32
    %acc = arith.addf %o, %mm : f32
    linalg.yield %acc : f32
  } -> tensor<8x8xf32>

  return %fc2 : tensor<8x8xf32>
}

// The [8x256] fc1 result is no longer a standalone value feeding fc2; instead
// the fc1 generic is tiled and fused inside an scf.for over the K_ff=256 axis,
// producing an [8 x 128] activation tile that is GELU'd and contracted in place.

// CHECK:       %[[F2:.*]] = linalg.fill ins(%{{.*}}) {{.*}} -> tensor<8x8xf32>
// CHECK:       %[[LOOP:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[F2]]) -> (tensor<8x8xf32>)
// CHECK:         %[[W1_TILE:.*]] = tensor.extract_slice %[[W1]][0, %[[K]]] [16, 128] [1, 1]
// CHECK:         %[[FILL_TILE:.*]] = linalg.fill{{.*}} -> tensor<8x128xf32>
// CHECK:         %[[FC1_TILE:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:      ins(%[[X]], %[[W1_TILE]] : tensor<8x16xf32>, tensor<16x128xf32>)
// CHECK-SAME:      outs(%[[FILL_TILE]] : tensor<8x128xf32>)
// CHECK:         %[[FC2_TILE:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:      ins(%[[FC1_TILE]], %{{.*}}, %{{.*}} : tensor<8x128xf32>, tensor<128xf32>, tensor<128x8xf32>)
// CHECK-SAME:      outs(%[[ACC]] : tensor<8x8xf32>)
// CHECK:           math.tanh
// CHECK:         scf.yield %[[FC2_TILE]] : tensor<8x8xf32>
// CHECK:       return %[[LOOP]]

// CHECK-NOT:   linalg.fill{{.*}}tensor<8x256xf32>

// -----

// A matmul chain WITHOUT a tanh-GELU epilogue must be left alone (this pass only
// targets the MLP fc1->GELU->fc2 shape).

#a = affine_map<(m, k, n) -> (m, k)>
#b = affine_map<(m, k, n) -> (k, n)>
#c = affine_map<(m, k, n) -> (m, n)>

// CHECK-LABEL: func.func @plain_matmul_chain
func.func @plain_matmul_chain(%x: tensor<8x16xf32>, %w1: tensor<16x256xf32>,
                              %w2: tensor<256x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %e1 = tensor.empty() : tensor<8x256xf32>
  %f1 = linalg.fill ins(%cst : f32) outs(%e1 : tensor<8x256xf32>) -> tensor<8x256xf32>
  // CHECK: linalg.generic
  // CHECK-NOT: scf.for
  %mm1 = linalg.generic {
      indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%x, %w1 : tensor<8x16xf32>, tensor<16x256xf32>)
      outs(%f1 : tensor<8x256xf32>) {
  ^bb0(%ai: f32, %bi: f32, %o: f32):
    %m = arith.mulf %ai, %bi : f32
    %s = arith.addf %o, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x256xf32>
  %e2 = tensor.empty() : tensor<8x8xf32>
  %f2 = linalg.fill ins(%cst : f32) outs(%e2 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %mm2 = linalg.generic {
      indexing_maps = [#a, #b, #c],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%mm1, %w2 : tensor<8x256xf32>, tensor<256x8xf32>)
      outs(%f2 : tensor<8x8xf32>) {
  ^bb0(%ai: f32, %bi: f32, %o: f32):
    %m = arith.mulf %ai, %bi : f32
    %s = arith.addf %o, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x8xf32>
  return %mm2 : tensor<8x8xf32>
}
