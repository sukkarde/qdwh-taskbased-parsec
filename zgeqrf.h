/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#ifndef ZGEQRF_H
#define ZGEQRF_H

dague_handle_t*
dplasma_zgeqrf_id_New( int optid,
                       tiled_matrix_desc_t *A1,
                       tiled_matrix_desc_t *A2,
                       tiled_matrix_desc_t *T2 );

void
dplasma_zgeqrf_id_Destruct( dague_handle_t *handle );

int
dplasma_zgeqrf_id( dague_context_t *dague,
                   int optid,
                   tiled_matrix_desc_t *A1,
                   tiled_matrix_desc_t *A2,
                   tiled_matrix_desc_t *T2 );

dague_handle_t*
dplasma_zgeqrf_split_New( int optqr, int optid,
                          tiled_matrix_desc_t *A1,
                          tiled_matrix_desc_t *A2,
                          tiled_matrix_desc_t *T1,
                          tiled_matrix_desc_t *T2 );

void
dplasma_zgeqrf_split_Destruct( dague_handle_t *handle );

int
dplasma_zgeqrf_split( dague_context_t *dague,
                      int optqr, int optid,
                      tiled_matrix_desc_t *A1,
                      tiled_matrix_desc_t *A2,
                      tiled_matrix_desc_t *T1,
                      tiled_matrix_desc_t *T2 );

dague_handle_t*
dplasma_zungqr_split_New( int optid,
                          tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                          tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                          tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

void
dplasma_zungqr_split_Destruct( dague_handle_t *handle );

int
dplasma_zungqr_split( dague_context_t *dague, int optid,
                      tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                      tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                      tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

dague_handle_t*
dplasma_zgeqrf_full_New( int optqr, int optid,
                         tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                         tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                         tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

void
dplasma_zgeqrf_full_Destruct( dague_handle_t *handle );

int
dplasma_zgeqrf_full( dague_context_t *dague, int optqr, int optid,
                     tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                     tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                     tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );
#endif
