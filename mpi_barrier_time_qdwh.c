/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @generated d Tue Feb  9 20:45:59 2016
 *
 */
#include <dague.h>
#include <dplasma.h>
#include <data_dist/matrix/two_dim_rectangle_cyclic.h>
#include <lapacke.h>
#include "flops.h"
#include "polar.h"
#include "dgeqrf.h"

extern double time_elapsed;
extern double sync_time_elapsed;

#if defined( DAGUE_HAVE_MPI)
# define get_cur_time() MPI_Wtime()
#else
static inline double get_cur_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv,NULL);
    t = tv.tv_sec + tv.tv_usec / 1e6;
    return t;
}
#endif


int    init = 0;
double eps;
double tol1;
double tol3;
int ib = 32;

int QDWH( int sym, int timing, int optqr, int optid,
          tiled_matrix_desc_t *descH, tiled_matrix_desc_t *descU,
          double *work, int ldw, int *Wi, double *flops, int *itQR, int *itPO,
          dague_context_t *dague )
{
    dague_handle_t *handle1, *handle2;
    two_dim_block_cyclic_t descB1, descB2, descQ1, descQ2, descT1, descT2;
    double *B1, *B2, *Q1, *Q2;
    double conv = 100.;
    double a, b, c, Liconv;
    double flops_dgeqrf, flops_dorgqr, flops_dgemm, flops_dpotrf, flops_dtrsm, id_ratio;
    double time, timer_qdwh, timer_condest, timer_H;
    double normest, Li, Unorm, Uinvnorm;
    size_t matsize1, matsize2;
    int M, N, MT, NT, nb, it, itconv, facto = -1;
    int info, nodes, rank, P;

    *itQR = 0;
    *itPO = 0;

    *flops = 0.0;
    //id_ratio = optid ? 0.5 : 1.;
    id_ratio = optid ? 0.5 : 1.5;

    if (!init) {
        eps  = LAPACKE_dlamch_work('e');
        tol1 = 5. * eps;
        tol3 = pow(tol1, 1./3.);
        init = 1;
    }

    M  = descH->m;
    N  = descH->n;
    MT = descH->mt;
    NT = descH->nt;
    nb = descH->mb;
    nodes = descH->super.nodes;
    rank  = descH->super.myrank;
    P     = ((two_dim_block_cyclic_t*)descH)->grid.rows;

    if ( M < N ){
        fprintf(stderr, "error(m >= n is required)") ;
        return -1;
    }

    /*  Start the timer */
    if(timing){
        time = polarGetTime();
        dague_context_wait( dague );
        timer_condest = -time;
    }

    /**
     * Let's initialize the 4 identical descriptors, and the two Ts
     */
    two_dim_block_cyclic_init( &descB1, matrix_RealDouble, matrix_Tile,
                               nodes, rank, nb, nb, M, N, 0, 0,
                               M, N, 1, 1, P );
    two_dim_block_cyclic_init( &descB2, matrix_RealDouble, matrix_Tile,
                               nodes, rank, nb, nb, N, N, 0, 0,
                               N, N, 1, 1, P );
    two_dim_block_cyclic_init( &descQ1, matrix_RealDouble, matrix_Tile,
                               nodes, rank, nb, nb, M, N, 0, 0,
                               M, N, 1, 1, P );
    two_dim_block_cyclic_init( &descQ2, matrix_RealDouble, matrix_Tile,
                               nodes, rank, nb, nb, N, N, 0, 0,
                               N, N, 1, 1, P );
    two_dim_block_cyclic_init( &descT1, matrix_RealDouble, matrix_Tile,
                               nodes, rank, ib, nb, MT*ib, N, 0, 0,
                               MT*ib, N, 1, 1, P );
    two_dim_block_cyclic_init( &descT2, matrix_RealDouble, matrix_Tile,
                               nodes, rank, ib, nb, NT*ib, N, 0, 0,
                               NT*ib, N, 1, 1, P );

    /**
     * Create the required workspaces
     */
    matsize1 = descB1.super.nb_local_tiles * descB1.super.bsiz;
    matsize2 = descB2.super.nb_local_tiles * descB2.super.bsiz;
    if ( work == NULL ) {
        matsize1 *= sizeof(double);
        B1 = (double *)malloc(matsize1);
        Q1 = (double *)malloc(matsize1);

        matsize2 *= sizeof(double);
        B2 = (double *)malloc(matsize2);
        Q2 = (double *)malloc(matsize2);
    }
    else {
        if( ldw < (2 * (matsize1 + matsize2)) ) {
            fprintf(stderr, "Providing workspace is too small: %d instead of %ld elements\n",
                    ldw, (2 * (matsize1 + matsize2)) );
            exit(-1);
        }
        B1 = work;
        B2 = B1 + matsize1;
        Q1 = B2 + matsize1;
        Q2 = Q1 + matsize2;
    }
    descB1.mat = B1;
    descB2.mat = B2;
    descQ1.mat = Q1;
    descQ2.mat = Q2;
    descT1.mat = malloc( descT1.super.nb_local_tiles * descT1.super.bsiz *
                         sizeof(double) );
    descT2.mat = malloc( descT2.super.nb_local_tiles * descT2.super.bsiz *
                         sizeof(double) );

    /**
     * Backup H in U
     */
    dague_context_start(dague);
    handle1 = dplasma_dlacpy_New( PlasmaUpperLower, descH, descU );
    dague_enqueue( dague, handle1 );
    /**
     * Two norm estimation
     */
    handle2 = dplasma_dlanm2_New( descH, &normest, &info );
    dague_enqueue( dague, handle2 );
    // if( sym )
    //  normest = dplasma_dsynm2( dague, descU );
    // else
    //  normest = dplasma_dgenm2( dague, descU );
    dague_context_wait(dague);
    dplasma_dlacpy_Destruct(handle1);
    dplasma_dlanm2_Destruct(handle2);

    /**
     * Scale the original U to form the U0 of the iterative loop
     */
    dplasma_dlascal( dague, PlasmaUpperLower,
                     1. / normest, descU );

    /**
     * Condition number estimation
     */
    dplasma_dlacpy( dague, PlasmaUpperLower, descU, (tiled_matrix_desc_t*)&descB1 );

    //if( optcond )
    {
        /**
        * Use QR for cond-est
        */
        dplasma_dgeqrf( dague,
                        (tiled_matrix_desc_t*)&descB1,
                        (tiled_matrix_desc_t*)&descT1 );

        dplasma_dlacpy( dague, PlasmaUpper,
                        (tiled_matrix_desc_t*)&descB1,
                        (tiled_matrix_desc_t*)&descB2 );
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time_elapsed = get_cur_time();
        dplasma_dtrtri( dague,
                        PlasmaUpper, PlasmaNonUnit,
                        (tiled_matrix_desc_t*)&descB2 );
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time_elapsed = get_cur_time() - sync_time_elapsed;
        //printf("\n sync_time_elapsed %e \n", sync_time_elapsed);
        Uinvnorm = dplasma_dlantr( dague, PlasmaOneNorm,
                                   PlasmaUpper, PlasmaNonUnit,
                                   (tiled_matrix_desc_t*)&descB2 );

        Li = ( 1.0 / Uinvnorm) / sqrt(N) ;
        *flops += FLOPS_DGEQRF( M, N )
                + FLOPS_DTRTRI( N );
    }
    //else 
    {
       /**
        * Use LU for cond-est
        */
        /*
        assert(0);
        // dplasma_dgetrf( dague, (tiled_matrix_desc_t*)&descB1, descIPIV );
         if ( sym )
             Unorm = dplasma_dlansy( dague, PlasmaOneNorm, PlasmaUpper, descU );
         else
             Unorm = dplasma_dlange( dague, PlasmaOneNorm, descU );

        // Li = dplasma_dgecon( dague, PlasmaOneNorm, (tiled_matrix_desc_t*)&descB1, Unorm );

        Li = Unorm / sqrt(N) * Li;
        *flops += FLOPS_DGETRF(N, N)
            + 2. * FLOPS_DTRSM( PlasmaLeft, N, 1 );
        */
        /**
         * WARNING: The cost of the gecon is estimated with only one iteration
         */
    }

    if(timing){
        dague_context_wait( dague );
        time = polarGetTime();
        timer_condest += time;
        timer_qdwh     = -time;
    }

    itconv = 0; Liconv = Li;
    while(itconv == 0 || fabs(1-Liconv) > tol1 ) {
        /* To find the minimum number of iterations to converge.
         * itconv = number of iterations needed until |Li - 1| < tol1
         * This should have converged in less than 50 iterations
         */
        if (itconv > 100) {
            exit(-1);
            break;
        }
        itconv++;

        Liconv = computeLi( Liconv, &a, &b, &c );
    }

    it = 0;
    //while(conv > tol3 || it == 0 || abs(1-Li) > tol1 ) {
    while(conv > tol3 || it < itconv ) {
        /* This should have converged in less than 50 iterations */
        if (it > 100) {
            exit(-1);
            break;
        }
        it++;

        Li = computeLi( Li, &a, &b, &c );
        fprintf(stderr, "Li = %e\n", Li);

        if ( c > 100 ) {
            int doqr = !optqr || it > 1;
            /**
             * Generate the matrix B = [ B1 ] = [ sqrt(c) * U ]
             *                         [ B2 ] = [ Id          ]
             */
            if( doqr ) {
                dplasma_dlacpy( dague, PlasmaUpperLower, descU, (tiled_matrix_desc_t*)&descB1 );
                dplasma_dlascal( dague, PlasmaUpperLower, sqrt(c), (tiled_matrix_desc_t*)&descB1 );
            }
            else {
                dplasma_dlascal( dague, PlasmaUpper, sqrt(c), (tiled_matrix_desc_t*)&descB1 );
            }

#if defined(MULTIPLE_JDF)
#if 1
            dplasma_dgeqrf_split( dague, !doqr, optid,
                                  (tiled_matrix_desc_t*)&descB1,
                                  (tiled_matrix_desc_t*)&descB2,
                                  (tiled_matrix_desc_t*)&descT1,
                                  (tiled_matrix_desc_t*)&descT2 );
#else
            if( doqr ) {
                dplasma_dgeqrf( dague,
                                (tiled_matrix_desc_t*)&descB1,
                                (tiled_matrix_desc_t*)&descT1 );
            }
            dplasma_dgeqrf_id( dague, optid,
                               (tiled_matrix_desc_t*)&descB1,
                               (tiled_matrix_desc_t*)&descB2,
                               (tiled_matrix_desc_t*)&descT2 );
#endif

            /**
             * Factorize B = QR, and generate the associated Q
             */
            dplasma_dorgqr_split( dague, optid,
                                  (tiled_matrix_desc_t*)&descB1, (tiled_matrix_desc_t*)&descB2,
                                  (tiled_matrix_desc_t*)&descT1, (tiled_matrix_desc_t*)&descT2,
                                  (tiled_matrix_desc_t*)&descQ1, (tiled_matrix_desc_t*)&descQ2 );

#else

            dplasma_dgeqrf_full( dague, !doqr, optid,
                                 (tiled_matrix_desc_t*)&descB1, (tiled_matrix_desc_t*)&descB2,
                                 (tiled_matrix_desc_t*)&descT1, (tiled_matrix_desc_t*)&descT2,
                                 (tiled_matrix_desc_t*)&descQ1, (tiled_matrix_desc_t*)&descQ2 );

#endif /* defined(MULTIPLE_JDF) */

            /* Cost of the upper part (B1) */
            flops_dgeqrf  = doqr ? FLOPS_DGEQRF( M, N ) : 0;
            /* Cost of the lower part (B2) */
            flops_dgeqrf += (optid ? 1./2. : 3./2.) * FLOPS_DGEQRF( N, N );

            //flops_dorgqr  = ( 1. + id_ratio ) * FLOPS_DORGQR( M, N, N );
            flops_dorgqr  = FLOPS_DORGQR( M, N, N ) + id_ratio * FLOPS_DORGQR( N, N, N );

            /**
             * Gemm to find the conv-norm
             *  U = ( (a-b/c)/sqrt(c) ) * Q1 * Q2' + (b/c) * U
             */
            if (it >= itconv ) {
                dplasma_dlacpy( dague, PlasmaUpperLower,
                                descU, (tiled_matrix_desc_t*)&descB2 );
            }

            dplasma_dgemm( dague, PlasmaNoTrans, PlasmaTrans,
                           ( (a-b/c)/sqrt(c) ), (tiled_matrix_desc_t*)&descQ1,
                                                (tiled_matrix_desc_t*)&descQ2,
                           (b/c),               descU );
            flops_dgemm  = FLOPS_DGEMM( M, N, N );

            /* Main flops used in this step */
            *flops += flops_dgeqrf + flops_dorgqr + flops_dgemm;

            facto = 0;
            *itQR = *itQR +1;
        }
        else {
            /**
             * Compute Q1 = c * U' * U + I
             * WARNING: This can't work in the general case, because Q1 is M-by-N while U'U is N-by-N
             */
            dplasma_dlaset( dague, PlasmaUpperLower, 0., 1.,
                            (tiled_matrix_desc_t*)&descQ1 );
            if (sym){
                dplasma_dsymm( dague, PlasmaLeft, PlasmaUpper,
                               c,   descU, descU,
                               1.0, (tiled_matrix_desc_t*)&descQ1 );
                flops_dgemm = FLOPS_DSYMM( PlasmaLeft, M, N );
            }
            else {
                //dplasma_dsyrk( dague, PlasmaUpper, PlasmaTrans,
                //               c,   descU, 
                //               1.0, (tiled_matrix_desc_t*)&descQ1 );
                //flops_dgemm = FLOPS_DSYRK( M, N);
                dplasma_dgemm( dague, PlasmaTrans, PlasmaNoTrans,
                               c,   descU, descU,
                               1.0, (tiled_matrix_desc_t*)&descQ1 );
                flops_dgemm = FLOPS_DGEMM( M, N, N );
            }

            /**
             * Solve Q2 x = Q1, with Q1 = U'
             * WARNING: This can't work in the general case, because U' is
             * N-by-M while Q2 is N-by-N, or Q1 is M-by-N. Need to split and
             * avoid using posv.
             */
            dplasma_dgeadd( dague, PlasmaTrans, 1.0, descU,
                            0.0, (tiled_matrix_desc_t*)&descQ2 );
            dplasma_dposv( dague, PlasmaUpper,
                           (tiled_matrix_desc_t*)&descQ1,
                           (tiled_matrix_desc_t*)&descQ2 );

            /**
             * Copy U to B1 to find conv-norm if it > itconv
             */
            if (it >= itconv ){
                dplasma_dlacpy( dague, PlasmaUpperLower,
                                descU, (tiled_matrix_desc_t*)&descB2 );
            }

            /**
             * Compute U =  (a-b/c) * Q2' + (b/c) * U
             */
            dplasma_dgeadd( dague, PlasmaTrans,
                            (a-b/c), (tiled_matrix_desc_t*)&descQ2,
                            (b/c),   descU );

            /* Main flops used in this step */
            flops_dpotrf = FLOPS_DPOTRF( M );
            flops_dtrsm  = FLOPS_DTRSM( PlasmaLeft, M, N );
            *flops += flops_dgemm + flops_dpotrf + 2. * flops_dtrsm;
            facto = 1;
            *itPO = *itPO +1;
        }

        if ( sym ) {
            /**
             * Symmetrize U using B2 as a workspace
             *      U = (U+U')/2 { B2 = U; U = 0.5 * ( U + B2' ); }
             */
            dplasma_dlacpy( dague, PlasmaUpperLower,
                            descU, (tiled_matrix_desc_t*)&descB1 );
            dplasma_dgeadd( dague, PlasmaTrans,
                            .5, (tiled_matrix_desc_t*)&descB1,
                            .5, descU );
        }

        /**
         * To find the conv-norm Compute the norm of the U - B1
         */
        conv = 10.;
        if(it >= itconv ){
            dplasma_dgeadd( dague, PlasmaNoTrans,
                             1.0, descU,
                            -1.0, (tiled_matrix_desc_t*)&descB2 );
            if( sym ) {
                conv = dplasma_dlansy( dague, PlasmaFrobeniusNorm,
                                       PlasmaUpper, (tiled_matrix_desc_t*)&descB2 );
            }
            else {
                conv = dplasma_dlange( dague, PlasmaFrobeniusNorm,
                                       (tiled_matrix_desc_t*)&descB2 );
            }
        }
        printf("%02d %-5s %e\n", it,
                facto == 0 ? "QR" : "PO", conv );
    }

    if(timing){
        dague_context_wait( dague );
        time = polarGetTime();
        timer_qdwh +=  time;
        timer_H     = -time;
    }

    /**
     * Compute H through the QDWH subroutine that
     * H = U'*A; H = .5(H+H')
     */
    if( sym ){
        dplasma_dsymm( dague, PlasmaRight, PlasmaUpper,
                       1., descU, descH,
                       0., (tiled_matrix_desc_t*)&descQ1 );
        flops_dgemm  = FLOPS_DSYMM( PlasmaLeft, M, N );
    }
    else {
        dplasma_dgemm( dague, PlasmaTrans, PlasmaNoTrans,
                       1., descU, descH,
                       0., (tiled_matrix_desc_t*)&descQ1 );
        flops_dgemm  = FLOPS_DGEMM( M, N, N );
    }

    *flops += FLOPS_DGEMM( M, N, N );

    dplasma_dlacpy( dague, PlasmaUpperLower,
                    (tiled_matrix_desc_t*)&descQ1, descH );

    dplasma_dgeadd( dague, PlasmaTrans,
                    .5, (tiled_matrix_desc_t*)&descQ1,
                    .5, descH );

    if(timing){
        dague_context_wait( dague );
        time = polarGetTime();
        timer_H  +=  time;
    }

    /* End the timer */
    dague_context_wait( dague );
    if(timing){
        fprintf(stderr, ": time  Cond-est QDWH H=U'A      \n"
               "===== %2.4e %2.4e %2.4e \n", timer_condest, timer_qdwh, timer_H);
    }
    fprintf(stderr, "\n");

    dague_data_free(descT1.mat);
    dague_data_free(descT2.mat);
    if ( work == NULL ) {
        free(B1);
        free(B2);
        free(Q1);
        free(Q2);
    }
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descB1 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descB2 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descQ1 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descQ2 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descT1 );
    tiled_matrix_desc_destroy( (tiled_matrix_desc_t*)&descT2 );
    return 0;
}
