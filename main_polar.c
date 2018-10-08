/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @generated d Tue Feb  9 20:45:59 2016
 *
 */

//make; mpirun -np 1 ./main_polar -c 35 -N 1800 -t 120 -T 120 -i 40 --verbose

#include <dague.h>
#include <dplasma.h>
#include <data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <lapacke.h>
#include "flops.h"
#include "polar.h"
#include "common.h"

int
dplasma_dlaset_sigma( dague_context_t *dague,
                      PLASMA_enum uplo,
                      double alpha,
                      double beta,
                      tiled_matrix_desc_t *A );

int QDWH( int sym, int timing, int optqr, int optid,
          tiled_matrix_desc_t *descH, tiled_matrix_desc_t *descU,
          double *work, int ldw, int *Wi, double *flops, int *itQR, int *itPO,
          dague_context_t *dague );

double cond = 1e16;

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int Aseed = 3872;
    double berr, orth;
    double Anorm;
    double flops = 0.0;
    int itQR = 0, itPO = 0;
    int sym = 0;
    //int optqr = 1;
    //int optid = 0;

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 32, 200, 200);

#if defined(DAGUE_HAVE_CUDA)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam, &cond);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    LDA = max(LDA, max(M, N));

    PASTE_CODE_ALLOCATE_MATRIX(descA, 1,
                               two_dim_block_cyclic, (&descA, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, M, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(descH, 1,
                               two_dim_block_cyclic, (&descH, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, M, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    PASTE_CODE_ALLOCATE_MATRIX(descU, 1,
                               two_dim_block_cyclic, (&descU, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, M, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    /*
    dplasma_dplrnt( dague, 0, (tiled_matrix_desc_t *)&descA, Aseed );
    if( sym ){
        dplasma_dlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&descA,
                        (tiled_matrix_desc_t *)&descH );
        dplasma_dgeadd(dague, PlasmaTrans,
                       .5, (tiled_matrix_desc_t *)&descH,
                       .5, (tiled_matrix_desc_t *)&descA );
    }
    */

   /*
    * generate matrix in a similar way to dlatms A = Q1 * Sigma * Q2
    * Sigma is a diagonal matrix of the desired singular values
    * Q1, Q2 are orthogonal matrices
    */

   /*
    * sets D(i)=1 - (i-1)/(N-1)*(1 - 1/COND)
    */
    //double cond = 1.e16;
    dplasma_dlaset_sigma( dague, PlasmaUpperLower, 0.0, cond, (tiled_matrix_desc_t *)&descA );
    double norm_sigma = dplasma_dlange( dague, PlasmaMaxNorm,
                                       (tiled_matrix_desc_t*)&descA );
    printf("\n norm_sigma %2.10e \n", norm_sigma);
   /*
    * generate two random matrices descH, descU
    * find qr(descH) and qr(descU)
    */

    PASTE_CODE_ALLOCATE_MATRIX(descT1, 1,
                               two_dim_block_cyclic, (&descT1, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(descQ1, 1,
                               two_dim_block_cyclic, (&descQ1, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, M, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(descT2, 1,
                               two_dim_block_cyclic, (&descT2, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(descQ2, 1,
                               two_dim_block_cyclic, (&descQ2, matrix_RealDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, M, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    dplasma_dplrnt( dague, 0, (tiled_matrix_desc_t *)&descH, Aseed );

    dplasma_dgeqrf( dague,
                    (tiled_matrix_desc_t*)&descH,
                    (tiled_matrix_desc_t*)&descT1 );
    /* dplasma_dlaset( dague, PlasmaUpperLower, 0., 1., */
    /*                 (tiled_matrix_desc_t*)&descQ1 ); */
    dplasma_dorgqr( dague,
                    (tiled_matrix_desc_t*)&descH,
                    (tiled_matrix_desc_t*)&descT1,
                    (tiled_matrix_desc_t*)&descQ1 );

    if( !sym) {
        dplasma_dplrnt( dague, 0, (tiled_matrix_desc_t *)&descU, 3800 );

        dplasma_dgeqrf( dague,
                        (tiled_matrix_desc_t*)&descU,
                        (tiled_matrix_desc_t*)&descT2 );
        /* dplasma_dlaset( dague, PlasmaUpperLower, 0., 1., */
        /*                 (tiled_matrix_desc_t*)&descQ2 ); */
        dplasma_dorgqr( dague,
                        (tiled_matrix_desc_t*)&descU,
                        (tiled_matrix_desc_t*)&descT2,
                        (tiled_matrix_desc_t*)&descQ2 );
    }

    dplasma_dgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                   1.0, (tiled_matrix_desc_t*)&descQ1,
                        (tiled_matrix_desc_t*)&descA,
                   0.0, (tiled_matrix_desc_t*)&descU );

    if( sym){
        dplasma_dgemm( dague, PlasmaNoTrans, PlasmaTrans,
                       1.0, (tiled_matrix_desc_t*)&descU,
                            (tiled_matrix_desc_t*)&descQ1,
                       0.0, (tiled_matrix_desc_t*)&descA );
    }
    else {
        dplasma_dgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                       1.0, (tiled_matrix_desc_t*)&descU,
                            (tiled_matrix_desc_t*)&descQ2,
                       0.0, (tiled_matrix_desc_t*)&descA );
    }

    dague_context_wait(dague);
    dague_data_free(descQ1.mat);
    dague_data_free(descT1.mat);
    dague_data_free(descQ2.mat);
    dague_data_free(descT2.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descQ1);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descT1 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descT2 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descQ2);

    Anorm = dplasma_dlange( dague, PlasmaFrobeniusNorm,
                            (tiled_matrix_desc_t *)&descA );

    /* Copy the matrix to check results */
    dplasma_dlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t *)&descA,
                    (tiled_matrix_desc_t *)&descH );

    /**
     * find polar decomposition A = UH: QDWH
     * U contains the orthogonal polar factor
     * H contains the hermitian polar factor
     */
    SYNC_TIME_START();
    TIME_START();
    QDWH( sym, (loud > 1), optqr, optid,
          (tiled_matrix_desc_t *)&descH,
          (tiled_matrix_desc_t *)&descU,
          NULL, 0, NULL, &flops, &itQR, &itPO,
          dague );
    SYNC_TIME_STOP();

    if(check){
        /* check the factorization |A-UH| */
        dplasma_dgemm( dague, PlasmaNoTrans, PlasmaNoTrans,
                       1., (tiled_matrix_desc_t *)&descU,
                           (tiled_matrix_desc_t *)&descH,
                      -1., (tiled_matrix_desc_t *)&descA );

        berr = dplasma_dlange( dague, PlasmaFrobeniusNorm,
                               (tiled_matrix_desc_t *)&descA );

        /* check the orthogonality |I-U'U| */
        dplasma_dlaset( dague, PlasmaUpperLower,
                        0., 1., (tiled_matrix_desc_t *)&descA);
        dplasma_dgemm( dague, PlasmaTrans, PlasmaNoTrans,
                        1., (tiled_matrix_desc_t *)&descU,
                            (tiled_matrix_desc_t *)&descU,
                       -1., (tiled_matrix_desc_t *)&descA );

        orth = dplasma_dlange( dague, PlasmaFrobeniusNorm,
                               (tiled_matrix_desc_t *)&descA );
    }
    else {
        berr = -1;
        orth = -1;
    }

    if ( rank == 0 ) {
        fprintf(stderr,
                "/////////////////////////////////////////////////////////////////////////\n"
                "# dplasma QDWH \n"
                "#\n"
                "# \tN  \t#itQR  \t#itPO  \tGflop/s \tMB   \tNB   \tIB   \tSMB   \tSNB   \tP   \tQ   \tTime    \tBerr  \t\tOrth \t\tCond \t\toptqr \toptid   \n"
                "   %6d \t%4d  \t%4d  \t%8.2f \t%4d \t%3d \t%3d \t%3d \t%3d \t%3d \t%3d \t%e \t%e \t%e \t%e \t%d \t%d\n"
                "/////////////////////////////////////////////////////////////////////////\n",
                M, itQR, itPO, flops/1e9/sync_time_elapsed, MB, NB, IB, SMB, SNB, P, Q, sync_time_elapsed, berr/Anorm, orth/Anorm, cond, optqr, optid);
    }

    dague_data_free(descA.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descA);
    dague_data_free(descH.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descH);
    dague_data_free(descU.mat);
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descU);

    cleanup_dague(dague, iparam);

    return 0;
}
