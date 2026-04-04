// Token + Position embedding
// token_ids:  memref<32xi32>         (input token indices)
// tok_table:  memref<256x64xf32>     (token embedding table, vocab=256)
// pos_table:  memref<32x64xf32>      (position embedding table, max_seq=32)
// output:     memref<32x64xf32>      (tok_emb + pos_emb)
module {
  func.func @embedding(%token_ids: memref<32xi32>,
                       %tok_table: memref<256x64xf32>,
                       %pos_table: memref<32x64xf32>,
                       %output: memref<32x64xf32>)
      attributes {vortex.entry} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index

    // For each position i in [0, seq_len):
    //   tok_id = token_ids[i]
    //   output[i, j] = tok_table[tok_id, j] + pos_table[i, j]
    scf.for %i = %c0 to %c32 step %c1 {
      %tok_id_i32 = memref.load %token_ids[%i] : memref<32xi32>
      %tok_id = arith.index_cast %tok_id_i32 : i32 to index
      scf.for %j = %c0 to %c64 step %c1 {
        %tok_val = memref.load %tok_table[%tok_id, %j] : memref<256x64xf32>
        %pos_val = memref.load %pos_table[%i, %j] : memref<32x64xf32>
        %sum = arith.addf %tok_val, %pos_val : f32
        memref.store %sum, %output[%i, %j] : memref<32x64xf32>
      }
    }
    return
  }
}
