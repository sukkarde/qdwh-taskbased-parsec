/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @generated d Sat Aug 13 16:38:52 2016
 *
 */

#ifndef DGEQRF_H
#define DGEQRF_H

dague_handle_t*
dplasma_dgeqrf_id_New( int optid,
                       tiled_matrix_desc_t *A1,
                       tiled_matrix_desc_t *A2,
                       tiled_matrix_desc_t *T2 );

void
dplasma_dgeqrf_id_Destruct( dague_handle_t *handle );

int
dplasma_dgeqrf_id( dague_context_t *dague,
                   int optid,
                   tiled_matrix_desc_t *A1,
                   tiled_matrix_desc_t *A2,
                   tiled_matrix_desc_t *T2 );

dague_handle_t*
dplasma_dgeqrf_split_New( int optqr, int optid,
                          tiled_matrix_desc_t *A1,
                          tiled_matrix_desc_t *A2,
                          tiled_matrix_desc_t *T1,
                          tiled_matrix_desc_t *T2 );

void
dplasma_dgeqrf_split_Destruct( dague_handle_t *handle );

int
dplasma_dgeqrf_split( dague_context_t *dague,
                      int optqr, int optid,
                      tiled_matrix_desc_t *A1,
                      tiled_matrix_desc_t *A2,
                      tiled_matrix_desc_t *T1,
                      tiled_matrix_desc_t *T2 );

dague_handle_t*
dplasma_dorgqr_split_New( int optid,
                          tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                          tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                          tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

void
dplasma_dorgqr_split_Destruct( dague_handle_t *handle );

int
dplasma_dorgqr_split( dague_context_t *dague, int optid,
                      tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                      tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                      tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

dague_handle_t*
dplasma_dgeqrf_full_New( int optqr, int optid,
                         tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                         tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                         tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );

void
dplasma_dgeqrf_full_Destruct( dague_handle_t *handle );

int
dplasma_dgeqrf_full( dague_context_t *dague, int optqr, int optid,
                     tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2,
                     tiled_matrix_desc_t *T1, tiled_matrix_desc_t *T2,
                     tiled_matrix_desc_t *Q1, tiled_matrix_desc_t *Q2 );
#endif
