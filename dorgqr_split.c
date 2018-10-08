#include "dague.h"
#include "dague/debug.h"
#include "dague/scheduling.h"
#include "dague/mca/pins/pins.h"
#include "dague/remote_dep.h"
#include "dague/datarepo.h"
#include "dague/data.h"
#include "dague/mempool.h"
#include "dague/utils/output.h"
#if defined(DAGUE_PROF_GRAPHER)
#include "dague/dague_prof_grapher.h"
#endif  /* defined(DAGUE_PROF_GRAPHER) */
#if defined(DAGUE_HAVE_CUDA)
#include "dague/devices/cuda/dev_cuda.h"
extern int dague_cuda_output_stream;
#endif  /* defined(DAGUE_HAVE_CUDA) */
#include <alloca.h>

#define DAGUE_dorgqr_split_NB_FUNCTIONS 11
#define DAGUE_dorgqr_split_NB_DATA 6

typedef struct __dague_dorgqr_split_internal_handle_s __dague_dorgqr_split_internal_handle_t;
struct dague_dorgqr_split_internal_handle_s;

/** Predeclarations of the dague_function_t */
static const dague_function_t dorgqr_split_ungqr_dtsmqr2_in_A2;
static const dague_function_t dorgqr_split_ungqr_dtsmqr2_in_T2;
static const dague_function_t dorgqr_split_ungqr_dtsmqr2;
static const dague_function_t dorgqr_split_ungqr_dtsmqr_in_A1;
static const dague_function_t dorgqr_split_ungqr_dtsmqr_in_T1;
static const dague_function_t dorgqr_split_ungqr_dtsmqr1;
static const dague_function_t dorgqr_split_ungqr_dormqr_in_A1;
static const dague_function_t dorgqr_split_ungqr_dormqr_in_T1;
static const dague_function_t dorgqr_split_ungqr_dormqr1;
static const dague_function_t dorgqr_split_ungqr_dlaset2;
static const dague_function_t dorgqr_split_ungqr_dlaset1;
/** Predeclarations of the parameters */
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_V;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_V;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_C;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_A;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_T;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dlaset2_for_A;
static const dague_flow_t flow_of_dorgqr_split_ungqr_dlaset1_for_A;
#line 2 "dorgqr_split.jdf"
/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * Copyright (c) 2013-2016 Inria. All rights reserved.
 * $COPYRIGHT
 *
 *
 * @generated d Sat Aug 13 16:38:53 2016
 *
 */
#include <data_dist/matrix/matrix.h>
#include <dague/private_mempool.h>
#include <lapacke.h>
#include <core_blas.h>

#if defined(DAGUE_HAVE_RECURSIVE)
#include <data_dist/matrix/subtile.h>
#include <dague/recursive.h>
#endif

#if defined(DAGUE_HAVE_CUDA)
#include <cores/dplasma_dcores.h>
#endif  /* defined(DAGUE_HAVE_CUDA) */


#line 85 "dorgqr_split.c"
#include "dorgqr_split.h"

struct __dague_dorgqr_split_internal_handle_s {
 dague_dorgqr_split_handle_t super;
 volatile uint32_t sync_point;
 dague_execution_context_t* startup_queue;
  /* The ranges to compute the hash key */
  int ungqr_dtsmqr2_in_A2_k_range;
  int ungqr_dtsmqr2_in_A2_m_range;
  int ungqr_dtsmqr2_in_T2_k_range;
  int ungqr_dtsmqr2_in_T2_m_range;
  int ungqr_dtsmqr2_k_range;
  int ungqr_dtsmqr2_m_range;
  int ungqr_dtsmqr2_n_range;
  int ungqr_dtsmqr_in_A1_k_range;
  int ungqr_dtsmqr_in_A1_m_range;
  int ungqr_dtsmqr_in_T1_k_range;
  int ungqr_dtsmqr_in_T1_m_range;
  int ungqr_dtsmqr1_k_range;
  int ungqr_dtsmqr1_m_range;
  int ungqr_dtsmqr1_n_range;
  int ungqr_dormqr_in_A1_k_range;
  int ungqr_dormqr_in_T1_k_range;
  int ungqr_dormqr1_k_range;
  int ungqr_dormqr1_n_range;
  int ungqr_dlaset2_m_range;
  int ungqr_dlaset2_n_range;
  int ungqr_dlaset1_m_range;
  int ungqr_dlaset1_n_range;
  /* The list of data repositories  ungqr_dtsmqr2_in_A2  ungqr_dtsmqr2_in_T2  ungqr_dtsmqr2  ungqr_dtsmqr_in_A1  ungqr_dtsmqr_in_T1  ungqr_dtsmqr1  ungqr_dormqr_in_A1  ungqr_dormqr_in_T1  ungqr_dormqr1  ungqr_dlaset2  ungqr_dlaset1 */
  data_repo_t* repositories[11];
};

#if defined(DAGUE_PROF_TRACE)
static int dorgqr_split_profiling_array[2*DAGUE_dorgqr_split_NB_FUNCTIONS] = {-1};
#endif  /* defined(DAGUE_PROF_TRACE) */
/* Globals */
#define optid (__dague_handle->super.optid)
#define p_work (__dague_handle->super.p_work)
#define descA1 (__dague_handle->super.descA1)
#define descA2 (__dague_handle->super.descA2)
#define descT1 (__dague_handle->super.descT1)
#define descT2 (__dague_handle->super.descT2)
#define descQ1 (__dague_handle->super.descQ1)
#define descQ2 (__dague_handle->super.descQ2)
#define ib (__dague_handle->super.ib)
#define KT (__dague_handle->super.KT)
#define KT2 (__dague_handle->super.KT2)

/* Data Access Macros */
#define dataA2(dataA20,dataA21)  (((dague_ddesc_t*)__dague_handle->super.dataA2)->data_of((dague_ddesc_t*)__dague_handle->super.dataA2, (dataA20), (dataA21)))

#define dataT2(dataT20,dataT21)  (((dague_ddesc_t*)__dague_handle->super.dataT2)->data_of((dague_ddesc_t*)__dague_handle->super.dataT2, (dataT20), (dataT21)))

#define dataA1(dataA10,dataA11)  (((dague_ddesc_t*)__dague_handle->super.dataA1)->data_of((dague_ddesc_t*)__dague_handle->super.dataA1, (dataA10), (dataA11)))

#define dataT1(dataT10,dataT11)  (((dague_ddesc_t*)__dague_handle->super.dataT1)->data_of((dague_ddesc_t*)__dague_handle->super.dataT1, (dataT10), (dataT11)))

#define dataQ2(dataQ20,dataQ21)  (((dague_ddesc_t*)__dague_handle->super.dataQ2)->data_of((dague_ddesc_t*)__dague_handle->super.dataQ2, (dataQ20), (dataQ21)))

#define dataQ1(dataQ10,dataQ11)  (((dague_ddesc_t*)__dague_handle->super.dataQ1)->data_of((dague_ddesc_t*)__dague_handle->super.dataQ1, (dataQ10), (dataQ11)))


/* Functions Predicates */
#define ungqr_dtsmqr2_in_A2_pred(k, mmax, m) (((dague_ddesc_t*)(__dague_handle->super.dataA2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, k))
#define ungqr_dtsmqr2_in_T2_pred(k, mmax, m) (((dague_ddesc_t*)(__dague_handle->super.dataT2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataT2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataT2, m, k))
#define ungqr_dtsmqr2_pred(k, mmax, m, n) (((dague_ddesc_t*)(__dague_handle->super.dataQ2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataQ2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n))
#define ungqr_dtsmqr_in_A1_pred(k, m) (((dague_ddesc_t*)(__dague_handle->super.dataA1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, k))
#define ungqr_dtsmqr_in_T1_pred(k, m) (((dague_ddesc_t*)(__dague_handle->super.dataT1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataT1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, m, k))
#define ungqr_dtsmqr1_pred(k, m, n) (((dague_ddesc_t*)(__dague_handle->super.dataQ1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataQ1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n))
#define ungqr_dormqr_in_A1_pred(k) (((dague_ddesc_t*)(__dague_handle->super.dataA1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, k))
#define ungqr_dormqr_in_T1_pred(k) (((dague_ddesc_t*)(__dague_handle->super.dataT1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataT1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, k, k))
#define ungqr_dormqr1_pred(k, n) (((dague_ddesc_t*)(__dague_handle->super.dataQ1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataQ1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, k, n))
#define ungqr_dlaset2_pred(m, n, k, mmax) (((dague_ddesc_t*)(__dague_handle->super.dataQ2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataQ2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n))
#define ungqr_dlaset1_pred(m, n, k, mmax, cq) (((dague_ddesc_t*)(__dague_handle->super.dataQ1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataQ1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n))

/* Data Repositories */
#define ungqr_dtsmqr2_in_A2_repo (__dague_handle->repositories[10])
#define ungqr_dtsmqr2_in_T2_repo (__dague_handle->repositories[9])
#define ungqr_dtsmqr2_repo (__dague_handle->repositories[8])
#define ungqr_dtsmqr_in_A1_repo (__dague_handle->repositories[7])
#define ungqr_dtsmqr_in_T1_repo (__dague_handle->repositories[6])
#define ungqr_dtsmqr1_repo (__dague_handle->repositories[5])
#define ungqr_dormqr_in_A1_repo (__dague_handle->repositories[4])
#define ungqr_dormqr_in_T1_repo (__dague_handle->repositories[3])
#define ungqr_dormqr1_repo (__dague_handle->repositories[2])
#define ungqr_dlaset2_repo (__dague_handle->repositories[1])
#define ungqr_dlaset1_repo (__dague_handle->repositories[0])
/* Dependency Tracking Allocation Macro */
#define ALLOCATE_DEP_TRACKING(DEPS, vMIN, vMAX, vNAME, vSYMBOL, PREVDEP, FLAG)               \
do {                                                                                         \
  int _vmin = (vMIN);                                                                        \
  int _vmax = (vMAX);                                                                        \
  (DEPS) = (dague_dependencies_t*)calloc(1, sizeof(dague_dependencies_t) +                   \
                   (_vmax - _vmin) * sizeof(dague_dependencies_union_t));                    \
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocate %d spaces for loop %s (min %d max %d) 0x%p last_dep 0x%p",    \
           (_vmax - _vmin + 1), (vNAME), _vmin, _vmax, (void*)(DEPS), (void*)(PREVDEP));    \
  (DEPS)->flags = DAGUE_DEPENDENCIES_FLAG_ALLOCATED | (FLAG);                                \
  (DEPS)->symbol = (vSYMBOL);                                                                \
  (DEPS)->min = _vmin;                                                                       \
  (DEPS)->max = _vmax;                                                                       \
  (DEPS)->prev = (PREVDEP); /* chain them backward */                                        \
} while (0)

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };

static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };

/* Release dependencies output macro */
#if defined(DAGUE_DEBUG_NOISIER)
#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "thread %d VP %d explore deps from %s:%s to %s:%s (from rank %d to %d) base ptr %p",\
           (NULL != (EU) ? (EU)->th_id : -1), (NULL != (EU) ? (EU)->virtual_process->vp_id : -1),\
           DEPO, dague_snprintf_execution_context(tmp1, 128, (dague_execution_context_t*)(TASKO)),\
           DEPI, dague_snprintf_execution_context(tmp2, 128, (dague_execution_context_t*)(TASKI)), (RSRC), (RDST), (DATA));\
  } while(0)
#define ACQUIRE_FLOW(TASKI, DEPI, FUNO, DEPO, LOCALS, PTR)\
  do { \
    char tmp1[128], tmp2[128]; (void)tmp1; (void)tmp2;\
    DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "task %s acquires flow %s from %s %s data ptr %p",\
           dague_snprintf_execution_context(tmp1, 128, (dague_execution_context_t*)(TASKI)), (DEPI),\
           (DEPO), dague_snprintf_assignments(tmp2, 128, (FUNO), (assignment_t*)(LOCALS)), (PTR));\
  } while(0)
#else
#define RELEASE_DEP_OUTPUT(EU, DEPO, TASKO, DEPI, TASKI, RSRC, RDST, DATA)
#define ACQUIRE_FLOW(TASKI, DEPI, TASKO, DEPO, LOCALS, PTR)
#endif
static inline int dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dtsmqr2_in_A2 */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;

  (void)k;  (void)mmax;  (void)m;

 return optid ? k : descQ2.mt-1; 
#line 226 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dtsmqr2_in_T2 */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;

  (void)k;  (void)mmax;  (void)m;

 return optid ? k : descQ2.mt-1; 
#line 240 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dtsmqr2 */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  (void)k;  (void)mmax;  (void)m;  (void)n;

 return optid ? k : descQ2.mt-1; 
#line 255 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dlaset2 */
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;

  (void)m;  (void)n;  (void)k;  (void)mmax;

 return optid ? k : descQ2.mt-1; 
#line 270 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dlaset2 */
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;

  (void)m;  (void)n;  (void)k;  (void)mmax;

 return dague_imin(KT, n); 
#line 285 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dlaset1 */
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int cq = assignments->cq.value;

  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;

 return 0; /*((descQ1.mt == descA1.nt) & (m == (descQ1.mt-1)) & (n == (descQ1.mt-1)));*/ 
#line 301 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dlaset1 */
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int cq = assignments->cq.value;

  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;

 return optid ? k : descQ2.mt-1; 
#line 317 "dorgqr_split.c"
}

static inline int dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task ungqr_dlaset1 */
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int cq = assignments->cq.value;

  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;

 return dague_imin(KT,dague_imin(m, n)); 
#line 333 "dorgqr_split.c"
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr2_in_A2(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr2_in_A2_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr2_in_T2(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr2_in_T2_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr2(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = k;
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr2_k_range;
  __h += (n - __jdf2c_n_min) * __dague_handle->ungqr_dtsmqr2_k_range * __dague_handle->ungqr_dtsmqr2_m_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr_in_A1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = (k + 1);
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr_in_A1_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr_in_T1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = (k + 1);
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr_in_T1_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dtsmqr1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = (k + 1);
  const int n = assignments->n.value;
  int __jdf2c_n_min = k;
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->ungqr_dtsmqr1_k_range;
  __h += (n - __jdf2c_n_min) * __dague_handle->ungqr_dtsmqr1_k_range * __dague_handle->ungqr_dtsmqr1_m_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dormqr_in_A1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  __h += (k - __jdf2c_k_min);
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dormqr_in_T1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  __h += (k - __jdf2c_k_min);
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dormqr1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = k;
  __h += (k - __jdf2c_k_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->ungqr_dormqr1_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dlaset2(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = 0;
  const int k = assignments->k.value;
  (void)k;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  __h += (m - __jdf2c_m_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->ungqr_dlaset2_m_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_ungqr_dlaset1(const __dague_dorgqr_split_internal_handle_t *__dague_handle,
                          const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = 0;
  const int k = assignments->k.value;
  (void)k;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int cq = assignments->cq.value;
  (void)cq;
  __h += (m - __jdf2c_m_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->ungqr_dlaset1_m_range;
 (void)__dague_handle; return __h;
}

/******                              ungqr_dtsmqr2_in_A2                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax, .max = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *locals)
{
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return mmax;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m = { .name = "m", .context_index = 2, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr2_in_A2(__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dtsmqr2_in_A2(__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
      /** Flow of V */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep1_atline_345_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)mmax;
  (void)m;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA2;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, k) )
    return __ddesc->data_of(__ddesc, m, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep1_atline_345 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataA2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep1_atline_345_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V,
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep2_atline_346 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_V,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V = {
  .name               = "V",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep1_atline_345 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep2_atline_346 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)mmax;  (void)m;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data V */
    data.data   = this_task->data.V.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
      const int ungqr_dtsmqr2_k = k;
      if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr2_k;
        const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
        const int ungqr_dtsmqr2_m = m;
        if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr2_m;
          int ungqr_dtsmqr2_n;
        for( ungqr_dtsmqr2_n = k;ungqr_dtsmqr2_n <= (descQ2.nt - 1); ungqr_dtsmqr2_n+=1) {
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "V", this_task, "V", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V_dep2_atline_346, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr2_in_A2_repo, __jdf2c_hash_ungqr_dtsmqr2_in_A2(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_A2(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr2_in_A2_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.V.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.V.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.V.data_in) ) {  /* flow V */
    entry = NULL;
    chunk = dague_data_get_copy(dataA2(m, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.V.data_in   = chunk;   /* flow V */
      this_task->data.V.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dtsmqr2_in_A2:V has been forwarded");
    }
    this_task->data.V.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k;  (void)mmax;  (void)m; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)mmax;  (void)m;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)mmax;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gV = this_task->data.V.data_in;
  void *V = (NULL != gV) ? DAGUE_DATA_COPY_GET_PTR(gV) : NULL; (void)V;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eV = this_task->data.V.data_repo;
    if( (NULL != eV) && (eV->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eV->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, V);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dtsmqr2_in_A2 BODY                            -----*/

#line 349 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 864 "dorgqr_split.c"
/*-----                        END OF ungqr_dtsmqr2_in_A2 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr2_in_A2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;  (void)mmax;  (void)m;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr2_in_A2(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_A2(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr2_in_A2_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t assignments;
  int32_t  k, mmax, m;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    assignments.mmax.value = mmax = dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, &assignments);
    for( assignments.m.value = m = 0;
        assignments.m.value <= mmax;
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      if( !ungqr_dtsmqr2_in_A2_pred(assignments.k.value, assignments.mmax.value, assignments.m.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr2_in_A2_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr2_in_A2_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr2_in_A2_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      mmax = dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, &assignments);
      assignments.mmax.value = mmax;
      __jdf2c_m_start = 0;
      __jdf2c_m_end = mmax;
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        if( !ungqr_dtsmqr2_in_A2_pred(k, mmax, m) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2_in_A2);
  __dague_handle->super.super.dependencies_array[10] = dep;
  __dague_handle->repositories[10] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dtsmqr2_in_A2(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int mmax = this_task->locals.mmax.value;  /* retrieve value saved during the last iteration */
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    this_task->locals.mmax.value = mmax = dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, &this_task->locals);
    for(this_task->locals.m.value = m = 0;
        this_task->locals.m.value <= mmax;
        this_task->locals.m.value += 1, m = this_task->locals.m.value) {
      if( !ungqr_dtsmqr2_in_A2_pred(k, mmax, m) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, k);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2_in_A2.function_id];
      new_task->chore_id     = 0;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.mmax.value = this_task->locals.mmax.value;
      new_task->locals.m.value = this_task->locals.m.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.V.data_repo = NULL;
      new_task->data.V.data_in   = NULL;
      new_task->data.V.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr2_in_A2_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr2_in_A2 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr2_in_A2 = {
  .name = "ungqr_dtsmqr2_in_A2",
  .function_id = 10,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 3,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k, &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_k, &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_mmax, &symb_dorgqr_split_ungqr_dtsmqr2_in_A2_m, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr2_in_A2,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr2_in_A2_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr2_in_A2,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              ungqr_dtsmqr2_in_T2                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax, .max = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *locals)
{
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return mmax;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m = { .name = "m", .context_index = 2, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr2_in_T2(__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataT2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dtsmqr2_in_T2(__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
      /** Flow of T */
    __d = (dague_ddesc_t*)__dague_handle->super.dataT2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep1_atline_327_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)mmax;
  (void)m;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataT2;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, k) )
    return __ddesc->data_of(__ddesc, m, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep1_atline_327 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataT2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep1_atline_327_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T,
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep2_atline_328 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_T,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T = {
  .name               = "T",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep1_atline_327 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep2_atline_328 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)mmax;  (void)m;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataT2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT2, m, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
      const int ungqr_dtsmqr2_k = k;
      if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr2_k;
        const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
        const int ungqr_dtsmqr2_m = m;
        if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr2_m;
          int ungqr_dtsmqr2_n;
        for( ungqr_dtsmqr2_n = k;ungqr_dtsmqr2_n <= (descQ2.nt - 1); ungqr_dtsmqr2_n+=1) {
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T_dep2_atline_328, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr2_in_T2_repo, __jdf2c_hash_ungqr_dtsmqr2_in_T2(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_T2(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr2_in_T2_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
    chunk = dague_data_get_copy(dataT2(m, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dtsmqr2_in_T2:T has been forwarded");
    }
    this_task->data.T.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k;  (void)mmax;  (void)m; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)mmax;  (void)m;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)mmax;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dtsmqr2_in_T2 BODY                            -----*/

#line 331 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 1474 "dorgqr_split.c"
/*-----                        END OF ungqr_dtsmqr2_in_T2 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr2_in_T2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;  (void)mmax;  (void)m;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr2_in_T2(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_T2(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr2_in_T2_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t assignments;
  int32_t  k, mmax, m;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    assignments.mmax.value = mmax = dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, &assignments);
    for( assignments.m.value = m = 0;
        assignments.m.value <= mmax;
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      if( !ungqr_dtsmqr2_in_T2_pred(assignments.k.value, assignments.mmax.value, assignments.m.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr2_in_T2_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr2_in_T2_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr2_in_T2_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      mmax = dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, &assignments);
      assignments.mmax.value = mmax;
      __jdf2c_m_start = 0;
      __jdf2c_m_end = mmax;
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        if( !ungqr_dtsmqr2_in_T2_pred(k, mmax, m) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2_in_T2);
  __dague_handle->super.super.dependencies_array[9] = dep;
  __dague_handle->repositories[9] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dtsmqr2_in_T2(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int mmax = this_task->locals.mmax.value;  /* retrieve value saved during the last iteration */
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    this_task->locals.mmax.value = mmax = dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, &this_task->locals);
    for(this_task->locals.m.value = m = 0;
        this_task->locals.m.value <= mmax;
        this_task->locals.m.value += 1, m = this_task->locals.m.value) {
      if( !ungqr_dtsmqr2_in_T2_pred(k, mmax, m) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataT2)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataT2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT2, m, k);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2_in_T2.function_id];
      new_task->chore_id     = 0;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.mmax.value = this_task->locals.mmax.value;
      new_task->locals.m.value = this_task->locals.m.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.T.data_repo = NULL;
      new_task->data.T.data_in   = NULL;
      new_task->data.T.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr2_in_T2_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr2_in_T2 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr2_in_T2 = {
  .name = "ungqr_dtsmqr2_in_T2",
  .function_id = 9,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 3,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k, &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_k, &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_mmax, &symb_dorgqr_split_ungqr_dtsmqr2_in_T2_m, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr2_in_T2,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr2_in_T2_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr2_in_T2,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                ungqr_dtsmqr2                                  ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_dorgqr_split_ungqr_dtsmqr2_mmax_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dtsmqr2_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dtsmqr2_mmax_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_mmax, .max = &expr_of_symb_dorgqr_split_ungqr_dtsmqr2_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return mmax;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_m = { .name = "m", .context_index = 2, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return k;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ2.nt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr2_n = { .name = "n", .context_index = 3, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr2_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr2(__dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
  (void)n;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataQ2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int final_data_of_dorgqr_split_ungqr_dtsmqr2(__dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)mmax;
  (void)m;
  (void)n;
      /** Flow of A1 */
    /** Flow of A2 */
    if( (k == 0) ) {
        __d = (dague_ddesc_t*)__dague_handle->super.dataQ2;
        refs[__flow_nb].ddesc = __d;
        refs[__flow_nb].key = __d->data_key(__d, m, n);
        __flow_nb++;
    }
    /** Flow of V */
    /** Flow of T */

    return __flow_nb;
}

static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == mmax);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277,  /* (m == mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 0, /* dorgqr_split_ungqr_dlaset1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dlaset1_for_A,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m < mmax);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278,  /* (m < mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return ((m == 0) && (k == (descQ1.mt - 1)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279,  /* ((m == 0) && (k == (descQ1.mt - 1))) */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dorgqr_split_ungqr_dormqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return ((m == 0) && (k < (descQ1.mt - 1)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280,  /* ((m == 0) && (k < (descQ1.mt - 1))) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m > 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281,  /* (m > 0) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
  .dep_index = 2,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277,
 &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279,
 &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280,
 &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281 }
};

static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return ((k == KT) || (n == k));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283,  /* ((k == KT) || (n == k)) */
  .ctl_gather_nb = NULL,
  .function_id = 1, /* dorgqr_split_ungqr_dlaset2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dlaset2_for_A,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return ((k < KT) && (n > k));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284,  /* ((k < KT) && (n > k)) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
  .dep_index = 3,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k == 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285_fct }
};
static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)mmax;
  (void)m;
  (void)n;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ2;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285,  /* (k == 0) */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataQ2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285_direct_access,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k > 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286,  /* (k > 0) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
  .dep_index = 4,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2 = {
  .name               = "A2",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 1,
  .flow_datatype_mask = 0x8,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283,
 &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep3_atline_285,
 &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286 }
};

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_V_dep1_atline_288 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 10, /* dorgqr_split_ungqr_dtsmqr2_in_A2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_A2_for_V,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_V,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_V = {
  .name               = "V",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 2,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_V_dep1_atline_288 },
  .dep_out    = { NULL }
};

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_T_dep1_atline_289 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 9, /* dorgqr_split_ungqr_dtsmqr2_in_T2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_in_T2_for_T,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr2_for_T = {
  .name               = "T",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 3,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_T_dep1_atline_289 },
  .dep_out    = { NULL }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)mmax;  (void)m;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n);
#endif
  if( action_mask & 0x7 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( ((m == 0) && (k == (descQ1.mt - 1))) ) {
      __dague_dorgqr_split_ungqr_dormqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr1.function_id];
        const int ungqr_dormqr1_k = k;
        if( (ungqr_dormqr1_k >= (0)) && (ungqr_dormqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dormqr1_k;
          const int ungqr_dormqr1_n = n;
          if( (ungqr_dormqr1_n >= (ncc->locals.k.value)) && (ungqr_dormqr1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dormqr1_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "C", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep3_atline_279, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( ((m == 0) && (k < (descQ1.mt - 1))) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = k;
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = (descQ1.mt - 1);
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep4_atline_280, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  if( action_mask & 0x4 ) {
        if( (m > 0) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = (m - 1);
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep5_atline_281, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x18 ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
    /* action_mask & 0x8 goes to data dataQ2(m, n) */
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x10 ) {
        if( (k > 0) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = (k - 1);
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = m;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep4_atline_286, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  /* Flow of data V has only IN dependencies */
  /* Flow of data T has only IN dependencies */
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)mmax;  (void)m;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == mmax) ) {
      __dague_dorgqr_split_ungqr_dlaset1_task_t* ncc = (__dague_dorgqr_split_ungqr_dlaset1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dlaset1.function_id];
        const int ungqr_dlaset1_m = k;
        if( (ungqr_dlaset1_m >= (0)) && (ungqr_dlaset1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dlaset1_m;
          const int ungqr_dlaset1_n = n;
          if( (ungqr_dlaset1_n >= (0)) && (ungqr_dlaset1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dlaset1_n;
            const int ungqr_dlaset1_k = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, &ncc->locals);
            assert(&nc.locals[2].value == &ncc->locals.k.value);
            ncc->locals.k.value = ungqr_dlaset1_k;
            const int ungqr_dlaset1_mmax = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, &ncc->locals);
            assert(&nc.locals[3].value == &ncc->locals.mmax.value);
            ncc->locals.mmax.value = ungqr_dlaset1_mmax;
            const int ungqr_dlaset1_cq = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, &ncc->locals);
            assert(&nc.locals[4].value == &ncc->locals.cq.value);
            ncc->locals.cq.value = ungqr_dlaset1_cq;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep1_atline_277, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (m < mmax) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = (m + 1);
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1_dep2_atline_278, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0xc ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x4 ) {
        if( ((k == KT) || (n == k)) ) {
      __dague_dorgqr_split_ungqr_dlaset2_task_t* ncc = (__dague_dorgqr_split_ungqr_dlaset2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dlaset2.function_id];
        const int ungqr_dlaset2_m = m;
        if( (ungqr_dlaset2_m >= (0)) && (ungqr_dlaset2_m <= ((descQ2.mt - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dlaset2_m;
          const int ungqr_dlaset2_n = n;
          if( (ungqr_dlaset2_n >= (0)) && (ungqr_dlaset2_n <= ((descQ2.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dlaset2_n;
            const int ungqr_dlaset2_k = dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, &ncc->locals);
            assert(&nc.locals[2].value == &ncc->locals.k.value);
            ncc->locals.k.value = ungqr_dlaset2_k;
            const int ungqr_dlaset2_mmax = dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, &ncc->locals);
            assert(&nc.locals[3].value == &ncc->locals.mmax.value);
            ncc->locals.mmax.value = ungqr_dlaset2_mmax;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep1_atline_283, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x8 ) {
        if( ((k < KT) && (n > k)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = (k + 1);
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = m;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2_dep2_atline_284, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x10 ) {  /* Flow of Data V */
    data.data   = this_task->data.V.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x10 ) {
        __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2_in_A2.function_id];
      const int ungqr_dtsmqr2_in_A2_k = k;
      if( (ungqr_dtsmqr2_in_A2_k >= (0)) && (ungqr_dtsmqr2_in_A2_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr2_in_A2_k;
        const int ungqr_dtsmqr2_in_A2_mmax = dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = ungqr_dtsmqr2_in_A2_mmax;
        const int ungqr_dtsmqr2_in_A2_m = m;
        if( (ungqr_dtsmqr2_in_A2_m >= (0)) && (ungqr_dtsmqr2_in_A2_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr2_in_A2_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority;
        RELEASE_DEP_OUTPUT(eu, "V", this_task, "V", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_V_dep1_atline_288, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  if( action_mask & 0x20 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x20 ) {
        __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2_in_T2.function_id];
      const int ungqr_dtsmqr2_in_T2_k = k;
      if( (ungqr_dtsmqr2_in_T2_k >= (0)) && (ungqr_dtsmqr2_in_T2_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr2_in_T2_k;
        const int ungqr_dtsmqr2_in_T2_mmax = dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = ungqr_dtsmqr2_in_T2_mmax;
        const int ungqr_dtsmqr2_in_T2_m = m;
        if( (ungqr_dtsmqr2_in_T2_m >= (0)) && (ungqr_dtsmqr2_in_T2_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr2_in_T2_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataT2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT2, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataT2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT2, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority;
        RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_T_dep1_atline_289, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr2_repo, __jdf2c_hash_ungqr_dtsmqr2(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr2(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr2_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    const int k = this_task->locals.k.value;
    const int mmax = this_task->locals.mmax.value;
    const int m = this_task->locals.m.value;
    const int n = this_task->locals.n.value;

    (void)k; (void)mmax; (void)m; (void)n;

    if( (m == mmax) ) {
      data_repo_entry_used_once( eu, ungqr_dlaset1_repo, this_task->data.A1.data_repo->key );
    }
    else if( (m < mmax) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr2_repo, this_task->data.A1.data_repo->key );
    }
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
    if( ((k == KT) || (n == k)) ) {
      data_repo_entry_used_once( eu, ungqr_dlaset2_repo, this_task->data.A2.data_repo->key );
    }
    else if( ((k < KT) && (n > k)) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr2_repo, this_task->data.A2.data_repo->key );
    }
    if( NULL != this_task->data.A2.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A2.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dtsmqr2_in_A2_repo, this_task->data.V.data_repo->key );
    if( NULL != this_task->data.V.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.V.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dtsmqr2_in_T2_repo, this_task->data.T.data_repo->key );
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A1.data_in) ) {  /* flow A1 */
    entry = NULL;
    if( (m == mmax) ) {
__dague_dorgqr_split_ungqr_dlaset1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dlaset1_assignment_t*)&generic_locals;
      const int ungqr_dlaset1m = target_locals->m.value = k; (void)ungqr_dlaset1m;
      const int ungqr_dlaset1n = target_locals->n.value = n; (void)ungqr_dlaset1n;
      const int ungqr_dlaset1k = target_locals->k.value = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, target_locals); (void)ungqr_dlaset1k;
      const int ungqr_dlaset1mmax = target_locals->mmax.value = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, target_locals); (void)ungqr_dlaset1mmax;
      const int ungqr_dlaset1cq = target_locals->cq.value = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, target_locals); (void)ungqr_dlaset1cq;
      entry = data_repo_lookup_entry( ungqr_dlaset1_repo, __jdf2c_hash_ungqr_dlaset1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:ungqr_dlaset1 to A1:ungqr_dtsmqr2");
      }
      chunk = entry->data[0];  /* A1:ungqr_dtsmqr2 <- A:ungqr_dlaset1 */
      ACQUIRE_FLOW(this_task, "A1", &dorgqr_split_ungqr_dlaset1, "A", target_locals, chunk);
    }
    else if( (m < mmax) ) {
__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr2k = target_locals->k.value = k; (void)ungqr_dtsmqr2k;
      const int ungqr_dtsmqr2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, target_locals); (void)ungqr_dtsmqr2mmax;
      const int ungqr_dtsmqr2m = target_locals->m.value = (m + 1); (void)ungqr_dtsmqr2m;
      const int ungqr_dtsmqr2n = target_locals->n.value = n; (void)ungqr_dtsmqr2n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr2_repo, __jdf2c_hash_ungqr_dtsmqr2( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:ungqr_dtsmqr2 to A1:ungqr_dtsmqr2");
      }
      chunk = entry->data[0];  /* A1:ungqr_dtsmqr2 <- A1:ungqr_dtsmqr2 */
      ACQUIRE_FLOW(this_task, "A1", &dorgqr_split_ungqr_dtsmqr2, "A1", target_locals, chunk);
    }
      this_task->data.A1.data_in   = chunk;   /* flow A1 */
      this_task->data.A1.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A1.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.A2.data_in) ) {  /* flow A2 */
    entry = NULL;
    if( ((k == KT) || (n == k)) ) {
__dague_dorgqr_split_ungqr_dlaset2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dlaset2_assignment_t*)&generic_locals;
      const int ungqr_dlaset2m = target_locals->m.value = m; (void)ungqr_dlaset2m;
      const int ungqr_dlaset2n = target_locals->n.value = n; (void)ungqr_dlaset2n;
      const int ungqr_dlaset2k = target_locals->k.value = dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, target_locals); (void)ungqr_dlaset2k;
      const int ungqr_dlaset2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, target_locals); (void)ungqr_dlaset2mmax;
      entry = data_repo_lookup_entry( ungqr_dlaset2_repo, __jdf2c_hash_ungqr_dlaset2( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:ungqr_dlaset2 to A2:ungqr_dtsmqr2");
      }
      chunk = entry->data[0];  /* A2:ungqr_dtsmqr2 <- A:ungqr_dlaset2 */
      ACQUIRE_FLOW(this_task, "A2", &dorgqr_split_ungqr_dlaset2, "A", target_locals, chunk);
    }
    else if( ((k < KT) && (n > k)) ) {
__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr2k = target_locals->k.value = (k + 1); (void)ungqr_dtsmqr2k;
      const int ungqr_dtsmqr2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, target_locals); (void)ungqr_dtsmqr2mmax;
      const int ungqr_dtsmqr2m = target_locals->m.value = m; (void)ungqr_dtsmqr2m;
      const int ungqr_dtsmqr2n = target_locals->n.value = n; (void)ungqr_dtsmqr2n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr2_repo, __jdf2c_hash_ungqr_dtsmqr2( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A2:ungqr_dtsmqr2 to A2:ungqr_dtsmqr2");
      }
      chunk = entry->data[1];  /* A2:ungqr_dtsmqr2 <- A2:ungqr_dtsmqr2 */
      ACQUIRE_FLOW(this_task, "A2", &dorgqr_split_ungqr_dtsmqr2, "A2", target_locals, chunk);
    }
      this_task->data.A2.data_in   = chunk;   /* flow A2 */
      this_task->data.A2.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A2.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.V.data_in) ) {  /* flow V */
    entry = NULL;
__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t*)&generic_locals;
  const int ungqr_dtsmqr2_in_A2k = target_locals->k.value = k; (void)ungqr_dtsmqr2_in_A2k;
  const int ungqr_dtsmqr2_in_A2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_in_A2_inline_c_expr1_line_339(__dague_handle, target_locals); (void)ungqr_dtsmqr2_in_A2mmax;
  const int ungqr_dtsmqr2_in_A2m = target_locals->m.value = m; (void)ungqr_dtsmqr2_in_A2m;
    entry = data_repo_lookup_entry( ungqr_dtsmqr2_in_A2_repo, __jdf2c_hash_ungqr_dtsmqr2_in_A2( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from V:ungqr_dtsmqr2_in_A2 to V:ungqr_dtsmqr2");
    }
    chunk = entry->data[0];  /* V:ungqr_dtsmqr2 <- V:ungqr_dtsmqr2_in_A2 */
    ACQUIRE_FLOW(this_task, "V", &dorgqr_split_ungqr_dtsmqr2_in_A2, "V", target_locals, chunk);
      this_task->data.V.data_in   = chunk;   /* flow V */
      this_task->data.V.data_repo = entry;
    }
    this_task->data.V.data_out = NULL;  /* input only */

  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t*)&generic_locals;
  const int ungqr_dtsmqr2_in_T2k = target_locals->k.value = k; (void)ungqr_dtsmqr2_in_T2k;
  const int ungqr_dtsmqr2_in_T2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_in_T2_inline_c_expr2_line_321(__dague_handle, target_locals); (void)ungqr_dtsmqr2_in_T2mmax;
  const int ungqr_dtsmqr2_in_T2m = target_locals->m.value = m; (void)ungqr_dtsmqr2_in_T2m;
    entry = data_repo_lookup_entry( ungqr_dtsmqr2_in_T2_repo, __jdf2c_hash_ungqr_dtsmqr2_in_T2( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from T:ungqr_dtsmqr2_in_T2 to T:ungqr_dtsmqr2");
    }
    chunk = entry->data[0];  /* T:ungqr_dtsmqr2 <- T:ungqr_dtsmqr2_in_T2 */
    ACQUIRE_FLOW(this_task, "T", &dorgqr_split_ungqr_dtsmqr2_in_T2, "T", target_locals, chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    this_task->data.T.data_out = NULL;  /* input only */

  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataQ2;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataQ2))->data_key((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)k;  (void)mmax;  (void)m;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    if( ((*flow_mask) & 0x1U) && ((m == mmax) || (m < mmax)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x2U) && (((k == KT) || (n == k)) || ((k < KT) && (n > k))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x2U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x4U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x4U) */
if( (*flow_mask) & 0x8U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x8U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x8U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x7U ) {  /* Flow A1 */
    if( ((*flow_mask) & 0x7U) && (((m == 0) && (k == (descQ1.mt - 1))) || ((m == 0) && (k < (descQ1.mt - 1))) || (m > 0)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x7U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x7U) */
if( (*flow_mask) & 0x18U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x18U) && ((k == 0) || (k > 0)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x18U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x18U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)mmax;  (void)m;  (void)n;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)k;  (void)mmax;  (void)m;  (void)n;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;
  dague_data_copy_t *gA2 = this_task->data.A2.data_in;
  void *A2 = (NULL != gA2) ? DAGUE_DATA_COPY_GET_PTR(gA2) : NULL; (void)A2;
  dague_data_copy_t *gV = this_task->data.V.data_in;
  void *V = (NULL != gV) ? DAGUE_DATA_COPY_GET_PTR(gV) : NULL; (void)V;
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA1 = this_task->data.A1.data_repo;
    if( (NULL != eA1) && (eA1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA1->sim_exec_date;
    data_repo_entry_t *eA2 = this_task->data.A2.data_repo;
    if( (NULL != eA2) && (eA2->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA2->sim_exec_date;
    data_repo_entry_t *eV = this_task->data.V.data_repo;
    if( (NULL != eV) && (eV->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eV->sim_exec_date;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
    if ( NULL != gA1 ) {
      dague_data_transfer_ownership_to_copy( gA1->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
    if ( NULL != gA2 ) {
      dague_data_transfer_ownership_to_copy( gA2->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
  cache_buf_referenced(context->closest_cache, A2);
  cache_buf_referenced(context->closest_cache, V);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              ungqr_dtsmqr2 BODY                              -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 292 "dorgqr_split.jdf"
{
    int tempAkm  = (k == (descA1.mt-1)) ? (descA1.m - k * descA1.mb) : descA1.mb;
    int tempAkn  = (k == (descA1.nt-1)) ? (descA1.n - k * descA1.nb) : descA1.nb;
    int tempkmin = dague_imin( tempAkm, tempAkn );
    int tempmm   = (m == (descQ2.mt-1)) ? (descQ2.m - m * descQ2.mb) : descQ2.mb;
    int tempnn   = (n == (descQ2.nt-1)) ? (descQ2.n - n * descQ2.nb) : descQ2.nb;
    int ldam = BLKLDD( descA2, m );
    int ldqk = BLKLDD( descQ1, k );
    int ldqm = BLKLDD( descQ2, m );

#if !defined(DAGUE_DRY_RUN)
    void *p_elem_A = dague_private_memory_pop( p_work );

    CORE_dtsmqr(PlasmaLeft, PlasmaNoTrans,
                descQ1.mb, tempnn, tempmm, tempnn, tempkmin, ib,
                A1 /* dataQ1(k, n) */, ldqk,
                A2 /* dataQ2(m, n) */, ldqm,
                V  /* dataA2(m, k) */, ldam,
                T  /* dataT2(m, k) */, descT2.mb,
                p_elem_A, ib );

    dague_private_memory_push( p_work, p_elem_A );
#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 2904 "dorgqr_split.c"
/*-----                          END OF ungqr_dtsmqr2 BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A1.data_out ) {
    this_task->data.A1.data_out->version++;  /* A1 */
  }
  if ( NULL != this_task->data.A2.data_out ) {
    this_task->data.A2.data_out->version++;  /* A2 */
  }
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == 0) ) {
    if( this_task->data.A2.data_out->original != dataQ2(m, n) ) {
      dague_dep_data_description_t data;
      data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
      data.layout = data.arena->opaque_dtt;
      data.count  = 1;
      data.displ  = 0;
      assert( data.count > 0 );
      dague_remote_dep_memcpy(this_task->dague_handle,
                              dague_data_get_copy(dataQ2(m, n), 0),
                              this_task->data.A2.data_out, &data);
    }
  }
  (void)k;  (void)mmax;  (void)m;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr2(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr2(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1f,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr2_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr2_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t assignments;
  int32_t  k, mmax, m, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    assignments.mmax.value = mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &assignments);
    for( assignments.m.value = m = 0;
        assignments.m.value <= mmax;
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      for( assignments.n.value = n = k;
          assignments.n.value <= (descQ2.nt - 1);
          assignments.n.value += 1, n = assignments.n.value) {
        __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
        __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
        if( !ungqr_dtsmqr2_pred(assignments.k.value, assignments.mmax.value, assignments.m.value, assignments.n.value) ) continue;
        nb_tasks++;
      }
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr2_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr2_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->ungqr_dtsmqr2_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr2_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &assignments);
      assignments.mmax.value = mmax;
      __jdf2c_m_start = 0;
      __jdf2c_m_end = mmax;
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        __jdf2c_n_start = k;
        __jdf2c_n_end = (descQ2.nt - 1);
        __jdf2c_n_inc = 1;
        for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
          assignments.n.value = n;
          if( !ungqr_dtsmqr2_pred(k, mmax, m, n) ) continue;
          /* We did find one! Allocate the dependencies array. */
          if( dep == NULL ) {
            ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr2_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr2_m, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dorgqr_split_ungqr_dtsmqr2_n, dep->u.next[k-__jdf2c_k_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
          }
      }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2);
  __dague_handle->super.super.dependencies_array[8] = dep;
  __dague_handle->repositories[8] = data_repo_create_nothreadsafe(nb_tasks, 4);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr2_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr2 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr2 = {
  .name = "ungqr_dtsmqr2",
  .function_id = 8,
  .nb_flows = 4,
  .nb_parameters = 3,
  .nb_locals = 4,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr2_k, &symb_dorgqr_split_ungqr_dtsmqr2_m, &symb_dorgqr_split_ungqr_dtsmqr2_n, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr2_k, &symb_dorgqr_split_ungqr_dtsmqr2_mmax, &symb_dorgqr_split_ungqr_dtsmqr2_m, &symb_dorgqr_split_ungqr_dtsmqr2_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr2,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dorgqr_split_ungqr_dtsmqr2,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_V, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1, &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0xf,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr2,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr2_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr2,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dorgqr_split_ungqr_dtsmqr2,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr2,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr2,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr2,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr2,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              ungqr_dtsmqr_in_A1                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT2;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr_in_A1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k + 1);
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.mt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr_in_A1_m = { .name = "m", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_A1_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr_in_A1(__dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)m;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dtsmqr_in_A1(__dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)m;
      /** Flow of V */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep1_atline_257_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int m = assignments->m.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)m;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, k) )
    return __ddesc->data_of(__ddesc, m, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep1_atline_257 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataA1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep1_atline_257_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V,
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep2_atline_258 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_V,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V = {
  .name               = "V",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep1_atline_257 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep2_atline_258 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)m;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data V */
    data.data   = this_task->data.V.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
      const int ungqr_dtsmqr1_k = k;
      if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr1_k;
        const int ungqr_dtsmqr1_m = m;
        if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[1].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr1_m;
          int ungqr_dtsmqr1_n;
        for( ungqr_dtsmqr1_n = k;ungqr_dtsmqr1_n <= (descQ1.nt - 1); ungqr_dtsmqr1_n+=1) {
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "V", this_task, "V", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V_dep2_atline_258, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr_in_A1_repo, __jdf2c_hash_ungqr_dtsmqr_in_A1(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_A1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr_in_A1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.V.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.V.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.V.data_in) ) {  /* flow V */
    entry = NULL;
    chunk = dague_data_get_copy(dataA1(m, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.V.data_in   = chunk;   /* flow V */
      this_task->data.V.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dtsmqr_in_A1:V has been forwarded");
    }
    this_task->data.V.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k;  (void)m; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)m;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gV = this_task->data.V.data_in;
  void *V = (NULL != gV) ? DAGUE_DATA_COPY_GET_PTR(gV) : NULL; (void)V;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eV = this_task->data.V.data_repo;
    if( (NULL != eV) && (eV->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eV->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, V);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dtsmqr_in_A1 BODY                            -----*/

#line 261 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 3458 "dorgqr_split.c"
/*-----                        END OF ungqr_dtsmqr_in_A1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;  (void)m;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr_in_A1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr_in_A1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr_in_A1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t assignments;
  int32_t  k, m;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT2;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    for( assignments.m.value = m = (k + 1);
        assignments.m.value <= (descQ1.mt - 1);
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      if( !ungqr_dtsmqr_in_A1_pred(assignments.k.value, assignments.m.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr_in_A1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr_in_A1_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr_in_A1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT2;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      __jdf2c_m_start = (k + 1);
      __jdf2c_m_end = (descQ1.mt - 1);
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        if( !ungqr_dtsmqr_in_A1_pred(k, m) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr_in_A1_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr_in_A1_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr_in_A1);
  __dague_handle->super.super.dependencies_array[7] = dep;
  __dague_handle->repositories[7] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dtsmqr_in_A1(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT2;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    for(this_task->locals.m.value = m = (k + 1);
        this_task->locals.m.value <= (descQ1.mt - 1);
        this_task->locals.m.value += 1, m = this_task->locals.m.value) {
      if( !ungqr_dtsmqr_in_A1_pred(k, m) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, k);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr_in_A1.function_id];
      new_task->chore_id     = 0;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.m.value = this_task->locals.m.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.V.data_repo = NULL;
      new_task->data.V.data_in   = NULL;
      new_task->data.V.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr_in_A1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr_in_A1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr_in_A1 = {
  .name = "ungqr_dtsmqr_in_A1",
  .function_id = 7,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 2,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr_in_A1_k, &symb_dorgqr_split_ungqr_dtsmqr_in_A1_m, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr_in_A1_k, &symb_dorgqr_split_ungqr_dtsmqr_in_A1_m, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr_in_A1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr_in_A1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr_in_A1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              ungqr_dtsmqr_in_T1                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT2;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr_in_T1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k + 1);
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.mt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr_in_T1_m = { .name = "m", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr_in_T1_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr_in_T1(__dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)m;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataT1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dtsmqr_in_T1(__dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)m;
      /** Flow of T */
    __d = (dague_ddesc_t*)__dague_handle->super.dataT1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep1_atline_240_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int m = assignments->m.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)m;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataT1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, k) )
    return __ddesc->data_of(__ddesc, m, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep1_atline_240 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataT1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep1_atline_240_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T,
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep2_atline_241 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_T,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T = {
  .name               = "T",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep1_atline_240 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep2_atline_241 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)m;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataT1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, m, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
      const int ungqr_dtsmqr1_k = k;
      if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr1_k;
        const int ungqr_dtsmqr1_m = m;
        if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[1].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr1_m;
          int ungqr_dtsmqr1_n;
        for( ungqr_dtsmqr1_n = k;ungqr_dtsmqr1_n <= (descQ1.nt - 1); ungqr_dtsmqr1_n+=1) {
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T_dep2_atline_241, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr_in_T1_repo, __jdf2c_hash_ungqr_dtsmqr_in_T1(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_T1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr_in_T1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
    chunk = dague_data_get_copy(dataT1(m, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dtsmqr_in_T1:T has been forwarded");
    }
    this_task->data.T.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k;  (void)m; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)m;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dtsmqr_in_T1 BODY                            -----*/

#line 244 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 4039 "dorgqr_split.c"
/*-----                        END OF ungqr_dtsmqr_in_T1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;  (void)m;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr_in_T1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr_in_T1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr_in_T1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t assignments;
  int32_t  k, m;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT2;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    for( assignments.m.value = m = (k + 1);
        assignments.m.value <= (descQ1.mt - 1);
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      if( !ungqr_dtsmqr_in_T1_pred(assignments.k.value, assignments.m.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr_in_T1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr_in_T1_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr_in_T1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT2;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      __jdf2c_m_start = (k + 1);
      __jdf2c_m_end = (descQ1.mt - 1);
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        if( !ungqr_dtsmqr_in_T1_pred(k, m) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr_in_T1_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr_in_T1_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr_in_T1);
  __dague_handle->super.super.dependencies_array[6] = dep;
  __dague_handle->repositories[6] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dtsmqr_in_T1(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT2;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    for(this_task->locals.m.value = m = (k + 1);
        this_task->locals.m.value <= (descQ1.mt - 1);
        this_task->locals.m.value += 1, m = this_task->locals.m.value) {
      if( !ungqr_dtsmqr_in_T1_pred(k, m) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT1, m, k);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr_in_T1.function_id];
      new_task->chore_id     = 0;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.m.value = this_task->locals.m.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.T.data_repo = NULL;
      new_task->data.T.data_in   = NULL;
      new_task->data.T.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr_in_T1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr_in_T1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr_in_T1 = {
  .name = "ungqr_dtsmqr_in_T1",
  .function_id = 6,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 2,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr_in_T1_k, &symb_dorgqr_split_ungqr_dtsmqr_in_T1_m, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr_in_T1_k, &symb_dorgqr_split_ungqr_dtsmqr_in_T1_m, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr_in_T1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr_in_T1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr_in_T1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                ungqr_dtsmqr1                                  ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k + 1);
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.mt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr1_m = { .name = "m", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return k;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.nt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dtsmqr1_n = { .name = "n", .context_index = 2, .min = &minexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dtsmqr1_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dtsmqr1(__dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)m;
  (void)n;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataQ1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int final_data_of_dorgqr_split_ungqr_dtsmqr1(__dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)m;
  (void)n;
      /** Flow of A1 */
    /** Flow of A2 */
    if( (k == 0) ) {
        __d = (dague_ddesc_t*)__dague_handle->super.dataQ1;
        refs[__flow_nb].ddesc = __d;
        refs[__flow_nb].key = __d->data_key(__d, m, n);
        __flow_nb++;
    }
    /** Flow of V */
    /** Flow of T */

    return __flow_nb;
}

static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == (descQ1.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191,  /* (m == (descQ1.mt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m < (descQ1.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192,  /* (m < (descQ1.mt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == (k + 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193,  /* (m == (k + 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dorgqr_split_ungqr_dormqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m > (k + 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194,  /* (m > (k + 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191,
 &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193,
 &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194 }
};

static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return ((k == KT) || (n == k));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196,  /* ((k == KT) || (n == k)) */
  .ctl_gather_nb = NULL,
  .function_id = 0, /* dorgqr_split_ungqr_dlaset1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dlaset1_for_A,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (((k < KT) && (n > k)) && (m == (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197,  /* (((k < KT) && (n > k)) && (m == (k + 1))) */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dorgqr_split_ungqr_dormqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
  .dep_index = 3,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (((k < KT) && (n > k)) && (m > (k + 1)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198,  /* (((k < KT) && (n > k)) && (m > (k + 1))) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
  .dep_index = 4,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k == 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199_fct }
};
static dague_data_t *flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)m;
  (void)n;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199,  /* (k == 0) */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataQ1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199_direct_access,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k > 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200,  /* (k > 0) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
  .dep_index = 3,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2 = {
  .name               = "A2",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 1,
  .flow_datatype_mask = 0x4,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196,
 &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197,
 &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep4_atline_199,
 &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200 }
};

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_V_dep1_atline_202 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 7, /* dorgqr_split_ungqr_dtsmqr_in_A1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr_in_A1_for_V,
  .dep_index = 5,
  .dep_datatype_index = 5,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_V,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_V = {
  .name               = "V",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 2,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_V_dep1_atline_202 },
  .dep_out    = { NULL }
};

static const dep_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_T_dep1_atline_203 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 6, /* dorgqr_split_ungqr_dtsmqr_in_T1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr_in_T1_for_T,
  .dep_index = 6,
  .dep_datatype_index = 6,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dtsmqr1_for_T = {
  .name               = "T",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 3,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_T_dep1_atline_203 },
  .dep_out    = { NULL }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)m;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == (k + 1)) ) {
      __dague_dorgqr_split_ungqr_dormqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr1.function_id];
        const int ungqr_dormqr1_k = k;
        if( (ungqr_dormqr1_k >= (0)) && (ungqr_dormqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dormqr1_k;
          const int ungqr_dormqr1_n = n;
          if( (ungqr_dormqr1_n >= (ncc->locals.k.value)) && (ungqr_dormqr1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dormqr1_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "C", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep3_atline_193, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (m > (k + 1)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = k;
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = (m - 1);
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep4_atline_194, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0xc ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
    /* action_mask & 0x4 goes to data dataQ1(m, n) */
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x8 ) {
        if( (k > 0) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = (k - 1);
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = m;
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep5_atline_200, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  /* Flow of data V has only IN dependencies */
  /* Flow of data T has only IN dependencies */
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)m;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == (descQ1.mt - 1)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = 0;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep1_atline_191, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (m < (descQ1.mt - 1)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = k;
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = (m + 1);
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1_dep2_atline_192, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x1c ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x4 ) {
        if( ((k == KT) || (n == k)) ) {
      __dague_dorgqr_split_ungqr_dlaset1_task_t* ncc = (__dague_dorgqr_split_ungqr_dlaset1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dlaset1.function_id];
        const int ungqr_dlaset1_m = m;
        if( (ungqr_dlaset1_m >= (0)) && (ungqr_dlaset1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dlaset1_m;
          const int ungqr_dlaset1_n = n;
          if( (ungqr_dlaset1_n >= (0)) && (ungqr_dlaset1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dlaset1_n;
            const int ungqr_dlaset1_k = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, &ncc->locals);
            assert(&nc.locals[2].value == &ncc->locals.k.value);
            ncc->locals.k.value = ungqr_dlaset1_k;
            const int ungqr_dlaset1_mmax = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, &ncc->locals);
            assert(&nc.locals[3].value == &ncc->locals.mmax.value);
            ncc->locals.mmax.value = ungqr_dlaset1_mmax;
            const int ungqr_dlaset1_cq = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, &ncc->locals);
            assert(&nc.locals[4].value == &ncc->locals.cq.value);
            ncc->locals.cq.value = ungqr_dlaset1_cq;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep1_atline_196, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x8 ) {
        if( (((k < KT) && (n > k)) && (m == (k + 1))) ) {
      __dague_dorgqr_split_ungqr_dormqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr1.function_id];
        const int ungqr_dormqr1_k = (k + 1);
        if( (ungqr_dormqr1_k >= (0)) && (ungqr_dormqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dormqr1_k;
          const int ungqr_dormqr1_n = n;
          if( (ungqr_dormqr1_n >= (ncc->locals.k.value)) && (ungqr_dormqr1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dormqr1_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "C", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep2_atline_197, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x10 ) {
        if( (((k < KT) && (n > k)) && (m > (k + 1))) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = (k + 1);
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = m;
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2_dep3_atline_198, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x20 ) {  /* Flow of Data V */
    data.data   = this_task->data.V.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x20 ) {
        __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr_in_A1.function_id];
      const int ungqr_dtsmqr_in_A1_k = k;
      if( (ungqr_dtsmqr_in_A1_k >= (0)) && (ungqr_dtsmqr_in_A1_k <= (KT2)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr_in_A1_k;
        const int ungqr_dtsmqr_in_A1_m = m;
        if( (ungqr_dtsmqr_in_A1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr_in_A1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[1].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr_in_A1_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority;
        RELEASE_DEP_OUTPUT(eu, "V", this_task, "V", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_V_dep1_atline_202, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  if( action_mask & 0x40 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x40 ) {
        __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr_in_T1.function_id];
      const int ungqr_dtsmqr_in_T1_k = k;
      if( (ungqr_dtsmqr_in_T1_k >= (0)) && (ungqr_dtsmqr_in_T1_k <= (KT2)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dtsmqr_in_T1_k;
        const int ungqr_dtsmqr_in_T1_m = m;
        if( (ungqr_dtsmqr_in_T1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr_in_T1_m <= ((descQ1.mt - 1))) ) {
          assert(&nc.locals[1].value == &ncc->locals.m.value);
          ncc->locals.m.value = ungqr_dtsmqr_in_T1_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataT1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT1, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority;
        RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_T_dep1_atline_203, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dtsmqr1_repo, __jdf2c_hash_ungqr_dtsmqr1(__dague_handle, (__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dtsmqr1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dtsmqr1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    const int k = this_task->locals.k.value;
    const int m = this_task->locals.m.value;
    const int n = this_task->locals.n.value;

    (void)k; (void)m; (void)n;

    if( (m == (descQ1.mt - 1)) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr2_repo, this_task->data.A1.data_repo->key );
    }
    else if( (m < (descQ1.mt - 1)) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr1_repo, this_task->data.A1.data_repo->key );
    }
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
    if( ((k == KT) || (n == k)) ) {
      data_repo_entry_used_once( eu, ungqr_dlaset1_repo, this_task->data.A2.data_repo->key );
    }
    else if( (((k < KT) && (n > k)) && (m == (k + 1))) ) {
      data_repo_entry_used_once( eu, ungqr_dormqr1_repo, this_task->data.A2.data_repo->key );
    }
    else if( (((k < KT) && (n > k)) && (m > (k + 1))) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr1_repo, this_task->data.A2.data_repo->key );
    }
    if( NULL != this_task->data.A2.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A2.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dtsmqr_in_A1_repo, this_task->data.V.data_repo->key );
    if( NULL != this_task->data.V.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.V.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dtsmqr_in_T1_repo, this_task->data.T.data_repo->key );
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A1.data_in) ) {  /* flow A1 */
    entry = NULL;
    if( (m == (descQ1.mt - 1)) ) {
__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr2k = target_locals->k.value = k; (void)ungqr_dtsmqr2k;
      const int ungqr_dtsmqr2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, target_locals); (void)ungqr_dtsmqr2mmax;
      const int ungqr_dtsmqr2m = target_locals->m.value = 0; (void)ungqr_dtsmqr2m;
      const int ungqr_dtsmqr2n = target_locals->n.value = n; (void)ungqr_dtsmqr2n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr2_repo, __jdf2c_hash_ungqr_dtsmqr2( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:ungqr_dtsmqr2 to A1:ungqr_dtsmqr1");
      }
      chunk = entry->data[0];  /* A1:ungqr_dtsmqr1 <- A1:ungqr_dtsmqr2 */
      ACQUIRE_FLOW(this_task, "A1", &dorgqr_split_ungqr_dtsmqr2, "A1", target_locals, chunk);
    }
    else if( (m < (descQ1.mt - 1)) ) {
__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr1k = target_locals->k.value = k; (void)ungqr_dtsmqr1k;
      const int ungqr_dtsmqr1m = target_locals->m.value = (m + 1); (void)ungqr_dtsmqr1m;
      const int ungqr_dtsmqr1n = target_locals->n.value = n; (void)ungqr_dtsmqr1n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr1_repo, __jdf2c_hash_ungqr_dtsmqr1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:ungqr_dtsmqr1 to A1:ungqr_dtsmqr1");
      }
      chunk = entry->data[0];  /* A1:ungqr_dtsmqr1 <- A1:ungqr_dtsmqr1 */
      ACQUIRE_FLOW(this_task, "A1", &dorgqr_split_ungqr_dtsmqr1, "A1", target_locals, chunk);
    }
      this_task->data.A1.data_in   = chunk;   /* flow A1 */
      this_task->data.A1.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A1.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.A2.data_in) ) {  /* flow A2 */
    entry = NULL;
    if( ((k == KT) || (n == k)) ) {
__dague_dorgqr_split_ungqr_dlaset1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dlaset1_assignment_t*)&generic_locals;
      const int ungqr_dlaset1m = target_locals->m.value = m; (void)ungqr_dlaset1m;
      const int ungqr_dlaset1n = target_locals->n.value = n; (void)ungqr_dlaset1n;
      const int ungqr_dlaset1k = target_locals->k.value = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, target_locals); (void)ungqr_dlaset1k;
      const int ungqr_dlaset1mmax = target_locals->mmax.value = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, target_locals); (void)ungqr_dlaset1mmax;
      const int ungqr_dlaset1cq = target_locals->cq.value = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, target_locals); (void)ungqr_dlaset1cq;
      entry = data_repo_lookup_entry( ungqr_dlaset1_repo, __jdf2c_hash_ungqr_dlaset1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:ungqr_dlaset1 to A2:ungqr_dtsmqr1");
      }
      chunk = entry->data[0];  /* A2:ungqr_dtsmqr1 <- A:ungqr_dlaset1 */
      ACQUIRE_FLOW(this_task, "A2", &dorgqr_split_ungqr_dlaset1, "A", target_locals, chunk);
    }
    else if( (((k < KT) && (n > k)) && (m == (k + 1))) ) {
__dague_dorgqr_split_ungqr_dormqr1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dormqr1_assignment_t*)&generic_locals;
      const int ungqr_dormqr1k = target_locals->k.value = (k + 1); (void)ungqr_dormqr1k;
      const int ungqr_dormqr1n = target_locals->n.value = n; (void)ungqr_dormqr1n;
      entry = data_repo_lookup_entry( ungqr_dormqr1_repo, __jdf2c_hash_ungqr_dormqr1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from C:ungqr_dormqr1 to A2:ungqr_dtsmqr1");
      }
      chunk = entry->data[0];  /* A2:ungqr_dtsmqr1 <- C:ungqr_dormqr1 */
      ACQUIRE_FLOW(this_task, "A2", &dorgqr_split_ungqr_dormqr1, "C", target_locals, chunk);
    }
    else if( (((k < KT) && (n > k)) && (m > (k + 1))) ) {
__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr1k = target_locals->k.value = (k + 1); (void)ungqr_dtsmqr1k;
      const int ungqr_dtsmqr1m = target_locals->m.value = m; (void)ungqr_dtsmqr1m;
      const int ungqr_dtsmqr1n = target_locals->n.value = n; (void)ungqr_dtsmqr1n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr1_repo, __jdf2c_hash_ungqr_dtsmqr1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A2:ungqr_dtsmqr1 to A2:ungqr_dtsmqr1");
      }
      chunk = entry->data[1];  /* A2:ungqr_dtsmqr1 <- A2:ungqr_dtsmqr1 */
      ACQUIRE_FLOW(this_task, "A2", &dorgqr_split_ungqr_dtsmqr1, "A2", target_locals, chunk);
    }
      this_task->data.A2.data_in   = chunk;   /* flow A2 */
      this_task->data.A2.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A2.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.V.data_in) ) {  /* flow V */
    entry = NULL;
__dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t*)&generic_locals;
  const int ungqr_dtsmqr_in_A1k = target_locals->k.value = k; (void)ungqr_dtsmqr_in_A1k;
  const int ungqr_dtsmqr_in_A1m = target_locals->m.value = m; (void)ungqr_dtsmqr_in_A1m;
    entry = data_repo_lookup_entry( ungqr_dtsmqr_in_A1_repo, __jdf2c_hash_ungqr_dtsmqr_in_A1( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from V:ungqr_dtsmqr_in_A1 to V:ungqr_dtsmqr1");
    }
    chunk = entry->data[0];  /* V:ungqr_dtsmqr1 <- V:ungqr_dtsmqr_in_A1 */
    ACQUIRE_FLOW(this_task, "V", &dorgqr_split_ungqr_dtsmqr_in_A1, "V", target_locals, chunk);
      this_task->data.V.data_in   = chunk;   /* flow V */
      this_task->data.V.data_repo = entry;
    }
    this_task->data.V.data_out = NULL;  /* input only */

  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
__dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t*)&generic_locals;
  const int ungqr_dtsmqr_in_T1k = target_locals->k.value = k; (void)ungqr_dtsmqr_in_T1k;
  const int ungqr_dtsmqr_in_T1m = target_locals->m.value = m; (void)ungqr_dtsmqr_in_T1m;
    entry = data_repo_lookup_entry( ungqr_dtsmqr_in_T1_repo, __jdf2c_hash_ungqr_dtsmqr_in_T1( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from T:ungqr_dtsmqr_in_T1 to T:ungqr_dtsmqr1");
    }
    chunk = entry->data[0];  /* T:ungqr_dtsmqr1 <- T:ungqr_dtsmqr_in_T1 */
    ACQUIRE_FLOW(this_task, "T", &dorgqr_split_ungqr_dtsmqr_in_T1, "T", target_locals, chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    this_task->data.T.data_out = NULL;  /* input only */

  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataQ1))->data_key((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)k;  (void)m;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    if( ((*flow_mask) & 0x1U) && ((m == (descQ1.mt - 1)) || (m < (descQ1.mt - 1))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x2U) && (((k == KT) || (n == k)) || (((k < KT) && (n > k)) && (m == (k + 1))) || (((k < KT) && (n > k)) && (m > (k + 1)))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x2U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x4U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x4U) */
if( (*flow_mask) & 0x8U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x8U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x8U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow A1 */
    if( ((*flow_mask) & 0x3U) && ((m == (k + 1)) || (m > (k + 1))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x3U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x3U) */
if( (*flow_mask) & 0xcU ) {  /* Flow A2 */
    if( ((*flow_mask) & 0xcU) && ((k == 0) || (k > 0)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0xcU;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0xcU) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)m;  (void)n;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)k;  (void)m;  (void)n;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;
  dague_data_copy_t *gA2 = this_task->data.A2.data_in;
  void *A2 = (NULL != gA2) ? DAGUE_DATA_COPY_GET_PTR(gA2) : NULL; (void)A2;
  dague_data_copy_t *gV = this_task->data.V.data_in;
  void *V = (NULL != gV) ? DAGUE_DATA_COPY_GET_PTR(gV) : NULL; (void)V;
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA1 = this_task->data.A1.data_repo;
    if( (NULL != eA1) && (eA1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA1->sim_exec_date;
    data_repo_entry_t *eA2 = this_task->data.A2.data_repo;
    if( (NULL != eA2) && (eA2->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA2->sim_exec_date;
    data_repo_entry_t *eV = this_task->data.V.data_repo;
    if( (NULL != eV) && (eV->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eV->sim_exec_date;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
    if ( NULL != gA1 ) {
      dague_data_transfer_ownership_to_copy( gA1->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
    if ( NULL != gA2 ) {
      dague_data_transfer_ownership_to_copy( gA2->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
  cache_buf_referenced(context->closest_cache, A2);
  cache_buf_referenced(context->closest_cache, V);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              ungqr_dtsmqr1 BODY                              -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 206 "dorgqr_split.jdf"
{
    int tempAkm  = (k == (descA1.mt-1)) ? (descA1.m - k * descA1.mb) : descA1.mb;
    int tempAkn  = (k == (descA1.nt-1)) ? (descA1.n - k * descA1.nb) : descA1.nb;
    int tempkmin = dague_imin( tempAkm, tempAkn );
    int tempmm   = (m == (descQ1.mt-1)) ? (descQ1.m - m * descQ1.mb) : descQ1.mb;
    int tempnn   = (n == (descQ1.nt-1)) ? (descQ1.n - n * descQ1.nb) : descQ1.nb;
    int ldam = BLKLDD( descA1, m );
    int ldqk = BLKLDD( descQ1, k );
    int ldqm = BLKLDD( descQ1, m );

#if !defined(DAGUE_DRY_RUN)
    void *p_elem_A = dague_private_memory_pop( p_work );

    CORE_dtsmqr(PlasmaLeft, PlasmaNoTrans,
                descQ1.mb, tempnn, tempmm, tempnn, tempkmin, ib,
                A1 /* dataQ1(k, n) */, ldqk,
                A2 /* dataQ1(m, n) */, ldqm,
                V  /* dataA1(m, k) */, ldam,
                T  /* dataT1(m, k) */, descT1.mb,
                p_elem_A, ib );

    dague_private_memory_push( p_work, p_elem_A );
#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 5431 "dorgqr_split.c"
/*-----                          END OF ungqr_dtsmqr1 BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dtsmqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dtsmqr1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A1.data_out ) {
    this_task->data.A1.data_out->version++;  /* A1 */
  }
  if ( NULL != this_task->data.A2.data_out ) {
    this_task->data.A2.data_out->version++;  /* A2 */
  }
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == 0) ) {
    if( this_task->data.A2.data_out->original != dataQ1(m, n) ) {
      dague_dep_data_description_t data;
      data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
      data.layout = data.arena->opaque_dtt;
      data.count  = 1;
      data.displ  = 0;
      assert( data.count > 0 );
      dague_remote_dep_memcpy(this_task->dague_handle,
                              dague_data_get_copy(dataQ1(m, n), 0),
                              this_task->data.A2.data_out, &data);
    }
  }
  (void)k;  (void)m;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dtsmqr1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dtsmqr1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0xf,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dtsmqr1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dtsmqr1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t assignments;
  int32_t  k, m, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    for( assignments.m.value = m = (k + 1);
        assignments.m.value <= (descQ1.mt - 1);
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      for( assignments.n.value = n = k;
          assignments.n.value <= (descQ1.nt - 1);
          assignments.n.value += 1, n = assignments.n.value) {
        __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
        __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
        if( !ungqr_dtsmqr1_pred(assignments.k.value, assignments.m.value, assignments.n.value) ) continue;
        nb_tasks++;
      }
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dtsmqr1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dtsmqr1_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->ungqr_dtsmqr1_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dtsmqr1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      __jdf2c_m_start = (k + 1);
      __jdf2c_m_end = (descQ1.mt - 1);
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        __jdf2c_n_start = k;
        __jdf2c_n_end = (descQ1.nt - 1);
        __jdf2c_n_inc = 1;
        for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
          assignments.n.value = n;
          if( !ungqr_dtsmqr1_pred(k, m, n) ) continue;
          /* We did find one! Allocate the dependencies array. */
          if( dep == NULL ) {
            ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dtsmqr1_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dtsmqr1_m, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dorgqr_split_ungqr_dtsmqr1_n, dep->u.next[k-__jdf2c_k_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
          }
      }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr1);
  __dague_handle->super.super.dependencies_array[5] = dep;
  __dague_handle->repositories[5] = data_repo_create_nothreadsafe(nb_tasks, 4);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dtsmqr1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dtsmqr1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dtsmqr1 = {
  .name = "ungqr_dtsmqr1",
  .function_id = 5,
  .nb_flows = 4,
  .nb_parameters = 3,
  .nb_locals = 3,
  .params = { &symb_dorgqr_split_ungqr_dtsmqr1_k, &symb_dorgqr_split_ungqr_dtsmqr1_m, &symb_dorgqr_split_ungqr_dtsmqr1_n, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dtsmqr1_k, &symb_dorgqr_split_ungqr_dtsmqr1_m, &symb_dorgqr_split_ungqr_dtsmqr1_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dtsmqr1,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dorgqr_split_ungqr_dtsmqr1,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_V, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1, &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0xf,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dtsmqr1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dtsmqr1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dtsmqr1,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dorgqr_split_ungqr_dtsmqr1,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dtsmqr1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dtsmqr1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dtsmqr1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dtsmqr1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              ungqr_dormqr_in_A1                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dormqr_in_A1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_A1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int affinity_of_dorgqr_split_ungqr_dormqr_in_A1(__dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, k, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dormqr_in_A1(__dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
      /** Flow of A */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, k, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep1_atline_173_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, k, k) )
    return __ddesc->data_of(__ddesc, k, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep1_atline_173 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataA1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep1_atline_173_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A,
};
static const dep_t flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep2_atline_174 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dorgqr_split_ungqr_dormqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr1_for_A,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A = {
  .name               = "A",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep1_atline_173 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep2_atline_174 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LOWER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dormqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr1.function_id];
      const int ungqr_dormqr1_k = k;
      if( (ungqr_dormqr1_k >= (0)) && (ungqr_dormqr1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dormqr1_k;
        int ungqr_dormqr1_n;
      for( ungqr_dormqr1_n = k;ungqr_dormqr1_n <= (descQ1.nt - 1); ungqr_dormqr1_n+=1) {
          if( (ungqr_dormqr1_n >= (ncc->locals.k.value)) && (ungqr_dormqr1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dormqr1_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A_dep2_atline_174, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dormqr_in_A1_repo, __jdf2c_hash_ungqr_dormqr_in_A1(__dague_handle, (__dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dormqr_in_A1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dormqr_in_A1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.A.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
    chunk = dague_data_get_copy(dataA1(k, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dormqr_in_A1:A has been forwarded");
    }
    this_task->data.A.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LOWER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LOWER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  (void)k;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA = this_task->data.A.data_in;
  void *A = (NULL != gA) ? DAGUE_DATA_COPY_GET_PTR(gA) : NULL; (void)A;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA = this_task->data.A.data_repo;
    if( (NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dormqr_in_A1 BODY                            -----*/

#line 177 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 5942 "dorgqr_split.c"
/*-----                        END OF ungqr_dormqr_in_A1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dormqr_in_A1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dormqr_in_A1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dormqr_in_A1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dormqr_in_A1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t assignments;
  int32_t  k;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    if( !ungqr_dormqr_in_A1_pred(assignments.k.value) ) continue;
    nb_tasks++;
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dormqr_in_A1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dormqr_in_A1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dormqr_in_A1_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr_in_A1);
  __dague_handle->super.super.dependencies_array[4] = dep;
  __dague_handle->repositories[4] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dormqr_in_A1(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    if( !ungqr_dormqr_in_A1_pred(k) ) continue;
    if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of ) {
      vpid = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, k);
      assert(context->nb_vp >= vpid);
    }
    new_task = (__dague_dorgqr_split_ungqr_dormqr_in_A1_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
    new_task->status = DAGUE_TASK_STATUS_NONE;
    /* Copy only the valid elements from this_task to new_task one */
    new_task->dague_handle = this_task->dague_handle;
    new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr_in_A1.function_id];
    new_task->chore_id     = 0;
    new_task->locals.k.value = this_task->locals.k.value;
    DAGUE_LIST_ITEM_SINGLETON(new_task);
    new_task->priority = __dague_handle->super.super.priority;
    new_task->data.A.data_repo = NULL;
    new_task->data.A.data_in   = NULL;
    new_task->data.A.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
    {
      char tmp[128];
      DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
             dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
    }
#endif
    dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
    pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                   &new_task->super.list_item,
                                                   dague_execution_context_priority_comparator);
    nb_tasks++;
   after_insert_task:  /* we jump here just so that we have code after the label */
    if( nb_tasks > this_task->locals.reserved[0].value ) {
      if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
      __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
      pready_ring = NULL;
      total_nb_tasks += nb_tasks;
      nb_tasks = 0;
      if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
        return DAGUE_HOOK_RETURN_AGAIN;
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dormqr_in_A1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dormqr_in_A1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dormqr_in_A1 = {
  .name = "ungqr_dormqr_in_A1",
  .function_id = 4,
  .nb_flows = 1,
  .nb_parameters = 1,
  .nb_locals = 1,
  .params = { &symb_dorgqr_split_ungqr_dormqr_in_A1_k, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dormqr_in_A1_k, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dormqr_in_A1,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dormqr_in_A1,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dormqr_in_A1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dormqr_in_A1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dormqr_in_A1,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dormqr_in_A1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dormqr_in_A1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dormqr_in_A1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dormqr_in_A1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              ungqr_dormqr_in_T1                              ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dormqr_in_T1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dormqr_in_T1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int affinity_of_dorgqr_split_ungqr_dormqr_in_T1(__dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataT1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, k, k);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dormqr_in_T1(__dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
      /** Flow of T */
    __d = (dague_ddesc_t*)__dague_handle->super.dataT1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, k, k);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep1_atline_157_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataT1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, k, k) )
    return __ddesc->data_of(__ddesc, k, k);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep1_atline_157 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataT1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep1_atline_157_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T,
};
static const dep_t flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep2_atline_158 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dorgqr_split_ungqr_dormqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr1_for_T,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T = {
  .name               = "T",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep1_atline_157 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep2_atline_158 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataT1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, k, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      __dague_dorgqr_split_ungqr_dormqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr1.function_id];
      const int ungqr_dormqr1_k = k;
      if( (ungqr_dormqr1_k >= (0)) && (ungqr_dormqr1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dormqr1_k;
        int ungqr_dormqr1_n;
      for( ungqr_dormqr1_n = k;ungqr_dormqr1_n <= (descQ1.nt - 1); ungqr_dormqr1_n+=1) {
          if( (ungqr_dormqr1_n >= (ncc->locals.k.value)) && (ungqr_dormqr1_n <= ((descQ1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = ungqr_dormqr1_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T_dep2_atline_158, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
        }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dormqr_in_T1_repo, __jdf2c_hash_ungqr_dormqr_in_T1(__dague_handle, (__dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dormqr_in_T1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dormqr_in_T1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
    chunk = dague_data_get_copy(dataT1(k, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow ungqr_dormqr_in_T1:T has been forwarded");
    }
    this_task->data.T.data_out = dague_data_get_copy(chunk->original, target_device);
  /** No profiling information */
  (void)k; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  (void)k;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            ungqr_dormqr_in_T1 BODY                            -----*/

#line 161 "dorgqr_split.jdf"
{
    /* nothing */
}

#line 6448 "dorgqr_split.c"
/*-----                        END OF ungqr_dormqr_in_T1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dormqr_in_T1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)k;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dormqr_in_T1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dormqr_in_T1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dormqr_in_T1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t assignments;
  int32_t  k;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    if( !ungqr_dormqr_in_T1_pred(assignments.k.value) ) continue;
    nb_tasks++;
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dormqr_in_T1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dormqr_in_T1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dormqr_in_T1_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr_in_T1);
  __dague_handle->super.super.dependencies_array[3] = dep;
  __dague_handle->repositories[3] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dormqr_in_T1(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.k.value = k = 0;
      this_task->locals.k.value <= KT;
      this_task->locals.k.value += 1, k = this_task->locals.k.value) {
    if( !ungqr_dormqr_in_T1_pred(k) ) continue;
    if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of ) {
      vpid = ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT1, k, k);
      assert(context->nb_vp >= vpid);
    }
    new_task = (__dague_dorgqr_split_ungqr_dormqr_in_T1_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
    new_task->status = DAGUE_TASK_STATUS_NONE;
    /* Copy only the valid elements from this_task to new_task one */
    new_task->dague_handle = this_task->dague_handle;
    new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr_in_T1.function_id];
    new_task->chore_id     = 0;
    new_task->locals.k.value = this_task->locals.k.value;
    DAGUE_LIST_ITEM_SINGLETON(new_task);
    new_task->priority = __dague_handle->super.super.priority;
    new_task->data.T.data_repo = NULL;
    new_task->data.T.data_in   = NULL;
    new_task->data.T.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
    {
      char tmp[128];
      DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
             dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
    }
#endif
    dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
    pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                   &new_task->super.list_item,
                                                   dague_execution_context_priority_comparator);
    nb_tasks++;
   after_insert_task:  /* we jump here just so that we have code after the label */
    if( nb_tasks > this_task->locals.reserved[0].value ) {
      if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
      __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
      pready_ring = NULL;
      total_nb_tasks += nb_tasks;
      nb_tasks = 0;
      if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
        return DAGUE_HOOK_RETURN_AGAIN;
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dormqr_in_T1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dormqr_in_T1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dormqr_in_T1 = {
  .name = "ungqr_dormqr_in_T1",
  .function_id = 3,
  .nb_flows = 1,
  .nb_parameters = 1,
  .nb_locals = 1,
  .params = { &symb_dorgqr_split_ungqr_dormqr_in_T1_k, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dormqr_in_T1_k, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dormqr_in_T1,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dormqr_in_T1,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dormqr_in_T1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dormqr_in_T1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dormqr_in_T1,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dormqr_in_T1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dormqr_in_T1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dormqr_in_T1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dormqr_in_T1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                ungqr_dormqr1                                  ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dormqr1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dormqr1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dormqr1_k_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dormqr1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dormqr1_k, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dormqr1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return k;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dormqr1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dormqr1_n_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.nt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_n_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dormqr1_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dormqr1_n, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dormqr1_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dormqr1(__dague_dorgqr_split_ungqr_dormqr1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)n;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataQ1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, k, n);
  return 1;
}
static inline int final_data_of_dorgqr_split_ungqr_dormqr1(__dague_dorgqr_split_ungqr_dormqr1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)n;
      /** Flow of C */
    if( (k == 0) ) {
        __d = (dague_ddesc_t*)__dague_handle->super.dataQ1;
        refs[__flow_nb].ddesc = __d;
        refs[__flow_nb].key = __d->data_key(__d, k, n);
        __flow_nb++;
    }
    /** Flow of A */
    /** Flow of T */

    return __flow_nb;
}

static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k == (descQ1.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120,  /* (k == (descQ1.mt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k < (descQ1.mt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121,  /* (k < (descQ1.mt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A1,
  .dep_index = 3,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k == 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122_fct }
};
static dague_data_t *flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int n = assignments->n.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)n;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, k, n) )
    return __ddesc->data_of(__ddesc, k, n);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122,  /* (k == 0) */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataQ1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dormqr1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k > 0);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123,  /* (k > 0) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_C,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_C = {
  .name               = "C",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120,
 &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep3_atline_122,
 &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123 }
};

static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_A_dep1_atline_117 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 4, /* dorgqr_split_ungqr_dormqr_in_A1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr_in_A1_for_A,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_A,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_A = {
  .name               = "A",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 1,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dormqr1_for_A_dep1_atline_117 },
  .dep_out    = { NULL }
};

static const dep_t flow_of_dorgqr_split_ungqr_dormqr1_for_T_dep1_atline_118 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dorgqr_split_ungqr_dormqr_in_T1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dormqr_in_T1_for_T,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dormqr1_for_T,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dormqr1_for_T = {
  .name               = "T",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 2,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dormqr1_for_T_dep1_atline_118 },
  .dep_out    = { NULL }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, k, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data C */
    data.data   = this_task->data.C.data_out;
    /* action_mask & 0x1 goes to data dataQ1(k, n) */
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x2 ) {
        if( (k > 0) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = (k - 1);
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = k;
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "C", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep4_atline_123, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  /* Flow of data A has only IN dependencies */
  /* Flow of data T has only IN dependencies */
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, k, n);
#endif
  if( action_mask & 0xc ) {  /* Flow of Data C */
    data.data   = this_task->data.C.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x4 ) {
        if( (k == (descQ1.mt - 1)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = 0;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "C", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep1_atline_120, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  if( action_mask & 0x8 ) {
        if( (k < (descQ1.mt - 1)) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = k;
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = (k + 1);
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "C", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr1_for_C_dep2_atline_121, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x1 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LOWER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr_in_A1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr_in_A1.function_id];
      const int ungqr_dormqr_in_A1_k = k;
      if( (ungqr_dormqr_in_A1_k >= (0)) && (ungqr_dormqr_in_A1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dormqr_in_A1_k;
#if defined(DISTRIBUTED)
        rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.k.value);
        if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
          vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.k.value);
        nc.priority = __dague_handle->super.super.priority;
      RELEASE_DEP_OUTPUT(eu, "A", this_task, "A", &nc, rank_src, rank_dst, &data);
      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr1_for_A_dep1_atline_117, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
        }
  }
  }
  if( action_mask & 0x2 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x2 ) {
        __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t* ncc = (__dague_dorgqr_split_ungqr_dormqr_in_T1_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dormqr_in_T1.function_id];
      const int ungqr_dormqr_in_T1_k = k;
      if( (ungqr_dormqr_in_T1_k >= (0)) && (ungqr_dormqr_in_T1_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = ungqr_dormqr_in_T1_k;
#if defined(DISTRIBUTED)
        rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataT1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataT1, ncc->locals.k.value, ncc->locals.k.value);
        if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
          vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataT1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataT1, ncc->locals.k.value, ncc->locals.k.value);
        nc.priority = __dague_handle->super.super.priority;
      RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
      if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dormqr1_for_T_dep1_atline_118, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dormqr1_repo, __jdf2c_hash_ungqr_dormqr1(__dague_handle, (__dague_dorgqr_split_ungqr_dormqr1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dormqr1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dormqr1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    const int k = this_task->locals.k.value;
    const int n = this_task->locals.n.value;

    (void)k; (void)n;

    if( (k == (descQ1.mt - 1)) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr2_repo, this_task->data.C.data_repo->key );
    }
    else if( (k < (descQ1.mt - 1)) ) {
      data_repo_entry_used_once( eu, ungqr_dtsmqr1_repo, this_task->data.C.data_repo->key );
    }
    if( NULL != this_task->data.C.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.C.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dormqr_in_A1_repo, this_task->data.A.data_repo->key );
    if( NULL != this_task->data.A.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A.data_in);
    }
    data_repo_entry_used_once( eu, ungqr_dormqr_in_T1_repo, this_task->data.T.data_repo->key );
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.C.data_in) ) {  /* flow C */
    entry = NULL;
    if( (k == (descQ1.mt - 1)) ) {
__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr2_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr2k = target_locals->k.value = k; (void)ungqr_dtsmqr2k;
      const int ungqr_dtsmqr2mmax = target_locals->mmax.value = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, target_locals); (void)ungqr_dtsmqr2mmax;
      const int ungqr_dtsmqr2m = target_locals->m.value = 0; (void)ungqr_dtsmqr2m;
      const int ungqr_dtsmqr2n = target_locals->n.value = n; (void)ungqr_dtsmqr2n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr2_repo, __jdf2c_hash_ungqr_dtsmqr2( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:ungqr_dtsmqr2 to C:ungqr_dormqr1");
      }
      chunk = entry->data[0];  /* C:ungqr_dormqr1 <- A1:ungqr_dtsmqr2 */
      ACQUIRE_FLOW(this_task, "C", &dorgqr_split_ungqr_dtsmqr2, "A1", target_locals, chunk);
    }
    else if( (k < (descQ1.mt - 1)) ) {
__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dtsmqr1_assignment_t*)&generic_locals;
      const int ungqr_dtsmqr1k = target_locals->k.value = k; (void)ungqr_dtsmqr1k;
      const int ungqr_dtsmqr1m = target_locals->m.value = (k + 1); (void)ungqr_dtsmqr1m;
      const int ungqr_dtsmqr1n = target_locals->n.value = n; (void)ungqr_dtsmqr1n;
      entry = data_repo_lookup_entry( ungqr_dtsmqr1_repo, __jdf2c_hash_ungqr_dtsmqr1( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:ungqr_dtsmqr1 to C:ungqr_dormqr1");
      }
      chunk = entry->data[0];  /* C:ungqr_dormqr1 <- A1:ungqr_dtsmqr1 */
      ACQUIRE_FLOW(this_task, "C", &dorgqr_split_ungqr_dtsmqr1, "A1", target_locals, chunk);
    }
      this_task->data.C.data_in   = chunk;   /* flow C */
      this_task->data.C.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.C.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
__dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t*)&generic_locals;
  const int ungqr_dormqr_in_A1k = target_locals->k.value = k; (void)ungqr_dormqr_in_A1k;
    entry = data_repo_lookup_entry( ungqr_dormqr_in_A1_repo, __jdf2c_hash_ungqr_dormqr_in_A1( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from A:ungqr_dormqr_in_A1 to A:ungqr_dormqr1");
    }
    chunk = entry->data[0];  /* A:ungqr_dormqr1 <- A:ungqr_dormqr_in_A1 */
    ACQUIRE_FLOW(this_task, "A", &dorgqr_split_ungqr_dormqr_in_A1, "A", target_locals, chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    this_task->data.A.data_out = NULL;  /* input only */

  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
__dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t *target_locals = (__dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t*)&generic_locals;
  const int ungqr_dormqr_in_T1k = target_locals->k.value = k; (void)ungqr_dormqr_in_T1k;
    entry = data_repo_lookup_entry( ungqr_dormqr_in_T1_repo, __jdf2c_hash_ungqr_dormqr_in_T1( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from T:ungqr_dormqr_in_T1 to T:ungqr_dormqr1");
    }
    chunk = entry->data[0];  /* T:ungqr_dormqr1 <- T:ungqr_dormqr_in_T1 */
    ACQUIRE_FLOW(this_task, "T", &dorgqr_split_ungqr_dormqr_in_T1, "T", target_locals, chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    this_task->data.T.data_out = NULL;  /* input only */

  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataQ1))->data_key((dague_ddesc_t*)__dague_handle->super.dataQ1, k, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)k;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow C */
    if( ((*flow_mask) & 0x1U) && ((k == (descQ1.mt - 1)) || (k < (descQ1.mt - 1))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LOWER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x2U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x4U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x4U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow C */
    if( ((*flow_mask) & 0x3U) && ((k == 0) || (k > 0)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x3U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x3U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)k;  (void)n;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  (void)k;  (void)n;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gC = this_task->data.C.data_in;
  void *C = (NULL != gC) ? DAGUE_DATA_COPY_GET_PTR(gC) : NULL; (void)C;
  dague_data_copy_t *gA = this_task->data.A.data_in;
  void *A = (NULL != gA) ? DAGUE_DATA_COPY_GET_PTR(gA) : NULL; (void)A;
  dague_data_copy_t *gT = this_task->data.T.data_in;
  void *T = (NULL != gT) ? DAGUE_DATA_COPY_GET_PTR(gT) : NULL; (void)T;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eC = this_task->data.C.data_repo;
    if( (NULL != eC) && (eC->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eC->sim_exec_date;
    data_repo_entry_t *eA = this_task->data.A.data_repo;
    if( (NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA->sim_exec_date;
    data_repo_entry_t *eT = this_task->data.T.data_repo;
    if( (NULL != eT) && (eT->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eT->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
    if ( NULL != gC ) {
      dague_data_transfer_ownership_to_copy( gC->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, C);
  cache_buf_referenced(context->closest_cache, A);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              ungqr_dormqr1 BODY                              -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 126 "dorgqr_split.jdf"
{
    int tempAkm  = (k == (descA1.mt-1)) ? (descA1.m - k * descA1.mb) : descA1.mb;
    int tempAkn  = (k == (descA1.nt-1)) ? (descA1.n - k * descA1.nb) : descA1.nb;
    int tempkmin = dague_imin( tempAkm, tempAkn );
    int tempkm   = (k == (descQ1.mt-1)) ? (descQ1.m - k * descQ1.mb) : descQ1.mb;
    int tempnn   = (n == (descQ1.nt-1)) ? (descQ1.n - n * descQ1.nb) : descQ1.nb;
    int ldak = BLKLDD( descA1, k );
    int ldqk = BLKLDD( descQ1, k );

#if !defined(DAGUE_DRY_RUN)
    void *p_elem_A = dague_private_memory_pop( p_work );

    CORE_dormqr(PlasmaLeft, PlasmaNoTrans,
                tempkm, tempnn, tempkmin, ib,
                A /* dataA1(k, k) */, ldak,
                T /* dataT1(k, k) */, descT1.mb,
                C /* dataQ1(k, n) */, ldqk,
                p_elem_A, descT1.nb );

    dague_private_memory_push( p_work, p_elem_A );
#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 7382 "dorgqr_split.c"
/*-----                          END OF ungqr_dormqr1 BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dormqr1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dormqr1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.C.data_out ) {
    this_task->data.C.data_out->version++;  /* C */
  }
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( (k == 0) ) {
    if( this_task->data.C.data_out->original != dataQ1(k, n) ) {
      dague_dep_data_description_t data;
      data.data   = this_task->data.C.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
      data.layout = data.arena->opaque_dtt;
      data.count  = 1;
      data.displ  = 0;
      assert( data.count > 0 );
      dague_remote_dep_memcpy(this_task->dague_handle,
                              dague_data_get_copy(dataQ1(k, n), 0),
                              this_task->data.C.data_out, &data);
    }
  }
  (void)k;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dormqr1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dormqr1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x3,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dormqr1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dormqr1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dormqr1_assignment_t assignments;
  int32_t  k, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    for( assignments.n.value = n = k;
        assignments.n.value <= (descQ1.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      if( !ungqr_dormqr1_pred(assignments.k.value, assignments.n.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dormqr1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->ungqr_dormqr1_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dormqr1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = KT;
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      __jdf2c_n_start = k;
      __jdf2c_n_end = (descQ1.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        if( !ungqr_dormqr1_pred(k, n) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dorgqr_split_ungqr_dormqr1_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dorgqr_split_ungqr_dormqr1_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr1);
  __dague_handle->super.super.dependencies_array[2] = dep;
  __dague_handle->repositories[2] = data_repo_create_nothreadsafe(nb_tasks, 3);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dormqr1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dormqr1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dormqr1 = {
  .name = "ungqr_dormqr1",
  .function_id = 2,
  .nb_flows = 3,
  .nb_parameters = 2,
  .nb_locals = 2,
  .params = { &symb_dorgqr_split_ungqr_dormqr1_k, &symb_dorgqr_split_ungqr_dormqr1_n, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dormqr1_k, &symb_dorgqr_split_ungqr_dormqr1_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dormqr1,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dorgqr_split_ungqr_dormqr1,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dormqr1_for_C, &flow_of_dorgqr_split_ungqr_dormqr1_for_A, &flow_of_dorgqr_split_ungqr_dormqr1_for_T, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dormqr1_for_C, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x7,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dormqr1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dormqr1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dormqr1,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dorgqr_split_ungqr_dormqr1,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dormqr1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dormqr1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dormqr1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dormqr1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                ungqr_dlaset2                                  ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dlaset2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dlaset2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dlaset2_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ2.mt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset2_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dlaset2_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dlaset2_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dlaset2_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dlaset2_n_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ2.nt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_n_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset2_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dlaset2_n, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dlaset2_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dorgqr_split_ungqr_dlaset2_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dlaset2_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dlaset2_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset2_k = { .name = "k", .context_index = 2, .min = &expr_of_symb_dorgqr_split_ungqr_dlaset2_k, .max = &expr_of_symb_dorgqr_split_ungqr_dlaset2_k, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dorgqr_split_ungqr_dlaset2_mmax_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dlaset2_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dlaset2_mmax_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset2_mmax = { .name = "mmax", .context_index = 3, .min = &expr_of_symb_dorgqr_split_ungqr_dlaset2_mmax, .max = &expr_of_symb_dorgqr_split_ungqr_dlaset2_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dlaset2(__dague_dorgqr_split_ungqr_dlaset2_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataQ2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dlaset2(__dague_dorgqr_split_ungqr_dlaset2_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
      /** Flow of A */
    __d = (dague_ddesc_t*)__dague_handle->super.dataQ2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, n);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep1_atline_92_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ2;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep1_atline_92 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataQ2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep1_atline_92_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dlaset2_for_A,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset2_assignment_t *locals)
{
  const int m = locals->m.value;
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return (m <= mmax);
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93,  /* (m <= mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A2,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dlaset2_for_A,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dlaset2_for_A = {
  .name               = "A",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep1_atline_92 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)m;  (void)n;  (void)k;  (void)mmax;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      if( (m <= mmax) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = m;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dlaset2_for_A_dep2_atline_93, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dlaset2_repo, __jdf2c_hash_ungqr_dlaset2(__dague_handle, (__dague_dorgqr_split_ungqr_dlaset2_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dlaset2(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dlaset2_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.A.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
    chunk = dague_data_get_copy(dataQ2(m, n), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** No profiling information */
  (void)m;  (void)n;  (void)k;  (void)mmax; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    if( ((*flow_mask) & 0x1U) && ((m <= mmax)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x1U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)m;  (void)n;  (void)k;  (void)mmax;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  (void)m;  (void)n;  (void)k;  (void)mmax;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA = this_task->data.A.data_in;
  void *A = (NULL != gA) ? DAGUE_DATA_COPY_GET_PTR(gA) : NULL; (void)A;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA = this_task->data.A.data_repo;
    if( (NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
    if ( NULL != gA ) {
      dague_data_transfer_ownership_to_copy( gA->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              ungqr_dlaset2 BODY                              -----*/

#line 96 "dorgqr_split.jdf"
{
    int tempmm = (m == (descQ2.mt-1)) ? (descQ2.m - m * descQ2.mb) : descQ2.mb;
    int tempnn = (n == (descQ2.nt-1)) ? (descQ2.n - n * descQ2.nb) : descQ2.nb;
    int ldqm = BLKLDD( descQ2, m );

#if !defined(DAGUE_DRY_RUN)
    CORE_dlaset(PlasmaUpperLower, tempmm, tempnn,
                      0., 0.,
                      A /* dataQ2(m,n) */, ldqm );
#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 7979 "dorgqr_split.c"
/*-----                          END OF ungqr_dlaset2 BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dlaset2(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A.data_out ) {
    this_task->data.A.data_out->version++;  /* A */
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)m;  (void)n;  (void)k;  (void)mmax;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dlaset2(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dlaset2(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dlaset2_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dlaset2_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dlaset2_assignment_t assignments;
  int32_t  m, n, k, mmax;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.m.value = m = 0;
      assignments.m.value <= (descQ2.mt - 1);
      assignments.m.value += 1, m = assignments.m.value) {
    __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
    __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
    for( assignments.n.value = n = 0;
        assignments.n.value <= (descQ2.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      assignments.k.value = k = dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, &assignments);
      assignments.mmax.value = mmax = dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, &assignments);
      if( !ungqr_dlaset2_pred(assignments.m.value, assignments.n.value, assignments.k.value, assignments.mmax.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dlaset2_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->ungqr_dlaset2_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dlaset2_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_m_start = 0;
    __jdf2c_m_end = (descQ2.mt - 1);
    __jdf2c_m_inc = 1;
    for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
      assignments.m.value = m;
      __jdf2c_n_start = 0;
      __jdf2c_n_end = (descQ2.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        k = dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, &assignments);
        assignments.k.value = k;
        mmax = dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, &assignments);
        assignments.mmax.value = mmax;
        if( !ungqr_dlaset2_pred(m, n, k, mmax) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dlaset2_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-__jdf2c_m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dorgqr_split_ungqr_dlaset2_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dlaset2);
  __dague_handle->super.super.dependencies_array[1] = dep;
  __dague_handle->repositories[1] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dlaset2(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dlaset2_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dlaset2_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  int n = this_task->locals.n.value;  /* retrieve value saved during the last iteration */
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int mmax = this_task->locals.mmax.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.m.value = m = 0;
      this_task->locals.m.value <= (descQ2.mt - 1);
      this_task->locals.m.value += 1, m = this_task->locals.m.value) {
    for(this_task->locals.n.value = n = 0;
        this_task->locals.n.value <= (descQ2.nt - 1);
        this_task->locals.n.value += 1, n = this_task->locals.n.value) {
      this_task->locals.k.value = k = dorgqr_split_ungqr_dlaset2_inline_c_expr5_line_86(__dague_handle, &this_task->locals);
      this_task->locals.mmax.value = mmax = dorgqr_split_ungqr_dlaset2_inline_c_expr4_line_87(__dague_handle, &this_task->locals);
      if( !ungqr_dlaset2_pred(m, n, k, mmax) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, m, n);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dlaset2_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dlaset2.function_id];
      new_task->chore_id     = 0;
      new_task->locals.m.value = this_task->locals.m.value;
      new_task->locals.n.value = this_task->locals.n.value;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.mmax.value = this_task->locals.mmax.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.A.data_repo = NULL;
      new_task->data.A.data_in   = NULL;
      new_task->data.A.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dlaset2_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dlaset2 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dlaset2 = {
  .name = "ungqr_dlaset2",
  .function_id = 1,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 4,
  .params = { &symb_dorgqr_split_ungqr_dlaset2_m, &symb_dorgqr_split_ungqr_dlaset2_n, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dlaset2_m, &symb_dorgqr_split_ungqr_dlaset2_n, &symb_dorgqr_split_ungqr_dlaset2_k, &symb_dorgqr_split_ungqr_dlaset2_mmax, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dlaset2,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dlaset2,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dlaset2_for_A, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dlaset2_for_A, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dlaset2,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dlaset2_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dlaset2,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dlaset2,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dlaset2,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dlaset2,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dlaset2,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                ungqr_dlaset1                                  ******/

static inline int minexpr_of_symb_dorgqr_split_ungqr_dlaset1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dlaset1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dlaset1_m_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_m_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.mt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_m_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset1_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_dorgqr_split_ungqr_dlaset1_m, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dorgqr_split_ungqr_dlaset1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dorgqr_split_ungqr_dlaset1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dorgqr_split_ungqr_dlaset1_n_fct }
};
static inline int maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_n_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descQ1.nt - 1);
}
static const expr_t maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_n_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset1_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_dorgqr_split_ungqr_dlaset1_n, .max = &maxexpr_of_symb_dorgqr_split_ungqr_dlaset1_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dorgqr_split_ungqr_dlaset1_k_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dlaset1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dlaset1_k_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset1_k = { .name = "k", .context_index = 2, .min = &expr_of_symb_dorgqr_split_ungqr_dlaset1_k, .max = &expr_of_symb_dorgqr_split_ungqr_dlaset1_k, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dorgqr_split_ungqr_dlaset1_mmax_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dlaset1_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dlaset1_mmax_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset1_mmax = { .name = "mmax", .context_index = 3, .min = &expr_of_symb_dorgqr_split_ungqr_dlaset1_mmax, .max = &expr_of_symb_dorgqr_split_ungqr_dlaset1_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dorgqr_split_ungqr_dlaset1_cq_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, locals);
}
static const expr_t expr_of_symb_dorgqr_split_ungqr_dlaset1_cq = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dorgqr_split_ungqr_dlaset1_cq_fct }
};
static const symbol_t symb_dorgqr_split_ungqr_dlaset1_cq = { .name = "cq", .context_index = 4, .min = &expr_of_symb_dorgqr_split_ungqr_dlaset1_cq, .max = &expr_of_symb_dorgqr_split_ungqr_dlaset1_cq, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dorgqr_split_ungqr_dlaset1(__dague_dorgqr_split_ungqr_dlaset1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
  (void)cq;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataQ1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int initial_data_of_dorgqr_split_ungqr_dlaset1(__dague_dorgqr_split_ungqr_dlaset1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
  (void)cq;
      /** Flow of A */
    __d = (dague_ddesc_t*)__dague_handle->super.dataQ1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, n);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep1_atline_59_direct_access(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int m = assignments->m.value;
  const int n = assignments->n.value;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int cq = assignments->cq.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)m;
  (void)n;
  (void)k;
  (void)mmax;
  (void)cq;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep1_atline_59 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dorgqr_split_dataQ1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep1_atline_59_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dlaset1_for_A,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  const int m = locals->m.value;
  const int n = locals->n.value;
  const int cq = locals->cq.value;

  (void)__dague_handle; (void)locals;
  return (!cq && ((m <= KT) && (n >= m)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61,  /* (!cq && ((m <= KT) && (n >= m))) */
  .ctl_gather_nb = NULL,
  .function_id = 8, /* dorgqr_split_ungqr_dtsmqr2 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr2_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dlaset1_for_A,
};
static inline int expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62_fct(const __dague_dorgqr_split_internal_handle_t *__dague_handle, const __dague_dorgqr_split_ungqr_dlaset1_assignment_t *locals)
{
  const int m = locals->m.value;
  const int n = locals->n.value;
  const int cq = locals->cq.value;

  (void)__dague_handle; (void)locals;
  return (!cq && ((m > KT) || (n < m)));
}
static const expr_t expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62_fct }
};
static const dep_t flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62 = {
  .cond = &expr_of_cond_for_flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62,  /* (!cq && ((m > KT) || (n < m))) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dorgqr_split_ungqr_dtsmqr1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dorgqr_split_ungqr_dtsmqr1_for_A2,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dorgqr_split_ungqr_dlaset1_for_A,
};
static const dague_flow_t flow_of_dorgqr_split_ungqr_dlaset1_for_A = {
  .name               = "A",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep1_atline_59 },
  .dep_out    = { &flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61,
 &flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62 }
};

static void
iterate_successors_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (!cq && ((m <= KT) && (n >= m))) ) {
      __dague_dorgqr_split_ungqr_dtsmqr2_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr2_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr2.function_id];
        const int ungqr_dtsmqr2_k = k;
        if( (ungqr_dtsmqr2_k >= (0)) && (ungqr_dtsmqr2_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr2_k;
          const int ungqr_dtsmqr2_mmax = dorgqr_split_ungqr_dtsmqr2_inline_c_expr3_line_270(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = ungqr_dtsmqr2_mmax;
          const int ungqr_dtsmqr2_m = mmax;
          if( (ungqr_dtsmqr2_m >= (0)) && (ungqr_dtsmqr2_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr2_m;
            const int ungqr_dtsmqr2_n = n;
            if( (ungqr_dtsmqr2_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr2_n <= ((descQ2.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr2_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep2_atline_61, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (!cq && ((m > KT) || (n < m))) ) {
      __dague_dorgqr_split_ungqr_dtsmqr1_task_t* ncc = (__dague_dorgqr_split_ungqr_dtsmqr1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dtsmqr1.function_id];
        const int ungqr_dtsmqr1_k = k;
        if( (ungqr_dtsmqr1_k >= (0)) && (ungqr_dtsmqr1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = ungqr_dtsmqr1_k;
          const int ungqr_dtsmqr1_m = m;
          if( (ungqr_dtsmqr1_m >= ((ncc->locals.k.value + 1))) && (ungqr_dtsmqr1_m <= ((descQ1.mt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.m.value);
            ncc->locals.m.value = ungqr_dtsmqr1_m;
            const int ungqr_dtsmqr1_n = n;
            if( (ungqr_dtsmqr1_n >= (ncc->locals.k.value)) && (ungqr_dtsmqr1_n <= ((descQ1.nt - 1))) ) {
              assert(&nc.locals[2].value == &ncc->locals.n.value);
              ncc->locals.n.value = ungqr_dtsmqr1_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority;
            RELEASE_DEP_OUTPUT(eu, "A", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dorgqr_split_ungqr_dlaset1_for_A_dep3_atline_62, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *eu, __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (const __dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  dague_release_dep_fct_arg_t arg;
  int __vp_id;
  arg.action_mask = action_mask;
  arg.output_usage = 0;
  arg.output_entry = NULL;
#if defined(DISTRIBUTED)
  arg.remote_deps = deps;
#endif  /* defined(DISTRIBUTED) */
  assert(NULL != eu);
  arg.ready_lists = alloca(sizeof(dague_execution_context_t *) * eu->virtual_process->dague_context->nb_vp);
  for( __vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; arg.ready_lists[__vp_id++] = NULL );
  (void)__dague_handle; (void)deps;
  if( action_mask & (DAGUE_ACTION_RELEASE_LOCAL_DEPS | DAGUE_ACTION_GET_REPO_ENTRY) ) {
    arg.output_entry = data_repo_lookup_entry_and_create( eu, ungqr_dlaset1_repo, __jdf2c_hash_ungqr_dlaset1(__dague_handle, (__dague_dorgqr_split_ungqr_dlaset1_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dorgqr_split_ungqr_dlaset1(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(ungqr_dlaset1_repo, arg.output_entry->key, arg.output_usage);
    for(__vp_id = 0; __vp_id < eu->virtual_process->dague_context->nb_vp; __vp_id++) {
      if( NULL == arg.ready_lists[__vp_id] ) continue;
      if(__vp_id == eu->virtual_process->vp_id) {
        __dague_schedule(eu, arg.ready_lists[__vp_id]);
      } else {
        __dague_schedule(vps[__vp_id]->execution_units[0], arg.ready_lists[__vp_id]);
      }
      arg.ready_lists[__vp_id] = NULL;
    }
  }
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    if( NULL != this_task->data.A.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
    chunk = dague_data_get_copy(dataQ1(m, n), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** No profiling information */
  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *eu, const __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow A */
    if( ((*flow_mask) & 0x3U) && ((!cq && ((m <= KT) && (n >= m))) || (!cq && ((m > KT) || (n < m)))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dorgqr_split_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x3U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x3U) */
 no_mask_match:
  data->arena  = NULL;
  data->data   = NULL;
  data->layout = DAGUE_DATATYPE_NULL;
  data->count  = 0;
  data->displ  = 0;
  (*flow_mask) = 0;  /* nothing left */
  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;
  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA = this_task->data.A.data_in;
  void *A = (NULL != gA) ? DAGUE_DATA_COPY_GET_PTR(gA) : NULL; (void)A;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA = this_task->data.A.data_repo;
    if( (NULL != eA) && (eA->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA->sim_exec_date;
    if( this_task->function->sim_cost_fct != NULL ) {
      this_task->sim_exec_date += this_task->function->sim_cost_fct(this_task);
    }
    if( context->largest_simulation_date < this_task->sim_exec_date )
      context->largest_simulation_date = this_task->sim_exec_date;
  }
#endif
  /** Transfer the ownership to the CPU */
#if defined(DAGUE_HAVE_CUDA)
    if ( NULL != gA ) {
      dague_data_transfer_ownership_to_copy( gA->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              ungqr_dlaset1 BODY                              -----*/

#line 65 "dorgqr_split.jdf"
{
    int tempmm = (m == (descQ1.mt-1)) ? (descQ1.m - m * descQ1.mb) : descQ1.mb;
    int tempnn = (n == (descQ1.nt-1)) ? (descQ1.n - n * descQ1.nb) : descQ1.nb;
    int ldqm = BLKLDD( descQ1, m );

#if !defined(DAGUE_DRY_RUN)
    CORE_dlaset(PlasmaUpperLower, tempmm, tempnn,
                0., (m == n) ? 1.: 0.,
                A /* dataQ(m,n) */, ldqm );
#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 8723 "dorgqr_split.c"
/*-----                          END OF ungqr_dlaset1 BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dorgqr_split_ungqr_dlaset1(dague_execution_unit_t *context, __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task)
{
  const __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int cq = this_task->locals.cq.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A.data_out ) {
    this_task->data.A.data_out->version++;  /* A */
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)m;  (void)n;  (void)k;  (void)mmax;  (void)cq;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_ungqr_dlaset1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dorgqr_split_ungqr_dlaset1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x3,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dorgqr_split_ungqr_dlaset1_internal_init(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dlaset1_task_t * this_task)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dorgqr_split_ungqr_dlaset1_assignment_t assignments;
  int32_t  m, n, k, mmax, cq;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.m.value = m = 0;
      assignments.m.value <= (descQ1.mt - 1);
      assignments.m.value += 1, m = assignments.m.value) {
    __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
    __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
    for( assignments.n.value = n = 0;
        assignments.n.value <= (descQ1.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      assignments.k.value = k = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, &assignments);
      assignments.mmax.value = mmax = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, &assignments);
      assignments.cq.value = cq = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, &assignments);
      if( !ungqr_dlaset1_pred(assignments.m.value, assignments.n.value, assignments.k.value, assignments.mmax.value, assignments.cq.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->ungqr_dlaset1_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->ungqr_dlaset1_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dorgqr_split_ungqr_dlaset1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_m_start = 0;
    __jdf2c_m_end = (descQ1.mt - 1);
    __jdf2c_m_inc = 1;
    for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
      assignments.m.value = m;
      __jdf2c_n_start = 0;
      __jdf2c_n_end = (descQ1.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        k = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, &assignments);
        assignments.k.value = k;
        mmax = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, &assignments);
        assignments.mmax.value = mmax;
        cq = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, &assignments);
        assignments.cq.value = cq;
        if( !ungqr_dlaset1_pred(m, n, k, mmax, cq) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dorgqr_split_ungqr_dlaset1_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-__jdf2c_m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dorgqr_split_ungqr_dlaset1_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
        }
    }
  }
  if( nb_tasks != DAGUE_UNDETERMINED_NB_TASKS ) {
    uint32_t ov, nv;
    do {
      ov = __dague_handle->super.super.nb_tasks;
      nv = (ov == DAGUE_UNDETERMINED_NB_TASKS ? DAGUE_UNDETERMINED_NB_TASKS : ov+nb_tasks);
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, nv) );
    nb_tasks = nv;
  } else {
    uint32_t ov;
    do {
      ov = __dague_handle->super.super.nb_tasks;
    } while( !dague_atomic_cas(&__dague_handle->super.super.nb_tasks, ov, DAGUE_UNDETERMINED_NB_TASKS));
  }
    do {
      this_task->super.list_item.list_next = (dague_list_item_t*)__dague_handle->startup_queue;
    } while(!dague_atomic_cas(&__dague_handle->startup_queue, this_task->super.list_item.list_next, this_task));
  } else this_task->status = DAGUE_TASK_STATUS_COMPLETE;

  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dlaset1);
  __dague_handle->super.super.dependencies_array[0] = dep;
  __dague_handle->repositories[0] = data_repo_create_nothreadsafe(nb_tasks, 1);
  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return (0 == nb_tasks) ? DAGUE_HOOK_RETURN_DONE : DAGUE_HOOK_RETURN_ASYNC;
}

static int __jdf2c_startup_ungqr_dlaset1(dague_execution_unit_t * eu, __dague_dorgqr_split_ungqr_dlaset1_task_t *this_task)
{
  __dague_dorgqr_split_ungqr_dlaset1_task_t* new_task;
  __dague_dorgqr_split_internal_handle_t* __dague_handle = (__dague_dorgqr_split_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  int n = this_task->locals.n.value;  /* retrieve value saved during the last iteration */
  int k = this_task->locals.k.value;  /* retrieve value saved during the last iteration */
  int mmax = this_task->locals.mmax.value;  /* retrieve value saved during the last iteration */
  int cq = this_task->locals.cq.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.m.value = m = 0;
      this_task->locals.m.value <= (descQ1.mt - 1);
      this_task->locals.m.value += 1, m = this_task->locals.m.value) {
    for(this_task->locals.n.value = n = 0;
        this_task->locals.n.value <= (descQ1.nt - 1);
        this_task->locals.n.value += 1, n = this_task->locals.n.value) {
      this_task->locals.k.value = k = dorgqr_split_ungqr_dlaset1_inline_c_expr8_line_52(__dague_handle, &this_task->locals);
      this_task->locals.mmax.value = mmax = dorgqr_split_ungqr_dlaset1_inline_c_expr7_line_53(__dague_handle, &this_task->locals);
      this_task->locals.cq.value = cq = dorgqr_split_ungqr_dlaset1_inline_c_expr6_line_54(__dague_handle, &this_task->locals);
      if( !ungqr_dlaset1_pred(m, n, k, mmax, cq) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataQ1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataQ1, m, n);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dorgqr_split_ungqr_dlaset1_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dorgqr_split_ungqr_dlaset1.function_id];
      new_task->chore_id     = 0;
      new_task->locals.m.value = this_task->locals.m.value;
      new_task->locals.n.value = this_task->locals.n.value;
      new_task->locals.k.value = this_task->locals.k.value;
      new_task->locals.mmax.value = this_task->locals.mmax.value;
      new_task->locals.cq.value = this_task->locals.cq.value;
      DAGUE_LIST_ITEM_SINGLETON(new_task);
      new_task->priority = __dague_handle->super.super.priority;
      new_task->data.A.data_repo = NULL;
      new_task->data.A.data_in   = NULL;
      new_task->data.A.data_out  = NULL;
#if defined(DAGUE_DEBUG_NOISIER)
      {
        char tmp[128];
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Add startup task %s",
               dague_snprintf_execution_context(tmp, 128, (dague_execution_context_t*)new_task));
      }
#endif
      dague_dependencies_mark_task_as_startup((dague_execution_context_t*)new_task);
      pready_ring = dague_list_item_ring_push_sorted(pready_ring,
                                                     &new_task->super.list_item,
                                                     dague_execution_context_priority_comparator);
      nb_tasks++;
     after_insert_task:  /* we jump here just so that we have code after the label */
      if( nb_tasks > this_task->locals.reserved[0].value ) {
        if( (size_t)this_task->locals.reserved[0].value < dague_task_startup_iter ) this_task->locals.reserved[0].value <<= 1;
        __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
        pready_ring = NULL;
        total_nb_tasks += nb_tasks;
        nb_tasks = 0;
        if( total_nb_tasks > dague_task_startup_chunk ) {  /* stop here and request to be rescheduled */
          return DAGUE_HOOK_RETURN_AGAIN;
        }
      }
    }
  }
  (void)eu; (void)vpid;
  if( NULL != pready_ring ) __dague_schedule(eu, (dague_execution_context_t*)pready_ring);
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dorgqr_split_ungqr_dlaset1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dorgqr_split_ungqr_dlaset1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dorgqr_split_ungqr_dlaset1 = {
  .name = "ungqr_dlaset1",
  .function_id = 0,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 5,
  .params = { &symb_dorgqr_split_ungqr_dlaset1_m, &symb_dorgqr_split_ungqr_dlaset1_n, NULL },
  .locals = { &symb_dorgqr_split_ungqr_dlaset1_m, &symb_dorgqr_split_ungqr_dlaset1_n, &symb_dorgqr_split_ungqr_dlaset1_k, &symb_dorgqr_split_ungqr_dlaset1_mmax, &symb_dorgqr_split_ungqr_dlaset1_cq, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dorgqr_split_ungqr_dlaset1,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dorgqr_split_ungqr_dlaset1,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dorgqr_split_ungqr_dlaset1_for_A, NULL },
  .out = { &flow_of_dorgqr_split_ungqr_dlaset1_for_A, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_ungqr_dlaset1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dorgqr_split_ungqr_dlaset1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dorgqr_split_ungqr_dlaset1,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dorgqr_split_ungqr_dlaset1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dorgqr_split_ungqr_dlaset1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dorgqr_split_ungqr_dlaset1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dorgqr_split_ungqr_dlaset1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


static const dague_function_t *dorgqr_split_functions[] = {
  &dorgqr_split_ungqr_dlaset1,
  &dorgqr_split_ungqr_dlaset2,
  &dorgqr_split_ungqr_dormqr1,
  &dorgqr_split_ungqr_dormqr_in_T1,
  &dorgqr_split_ungqr_dormqr_in_A1,
  &dorgqr_split_ungqr_dtsmqr1,
  &dorgqr_split_ungqr_dtsmqr_in_T1,
  &dorgqr_split_ungqr_dtsmqr_in_A1,
  &dorgqr_split_ungqr_dtsmqr2,
  &dorgqr_split_ungqr_dtsmqr2_in_T2,
  &dorgqr_split_ungqr_dtsmqr2_in_A2
};

static void dorgqr_split_startup(dague_context_t *context, __dague_dorgqr_split_internal_handle_t *__dague_handle, dague_list_item_t ** ready_tasks)
{
  uint32_t supported_dev = 0;
 
  uint32_t wanted_devices = __dague_handle->super.super.devices_mask; __dague_handle->super.super.devices_mask = 0;
  uint32_t _i;
  for( _i = 0; _i < dague_nb_devices; _i++ ) {
    if( !(wanted_devices & (1<<_i)) ) continue;
    dague_device_t* device = dague_devices_get(_i);
    dague_ddesc_t* dague_ddesc;
 
    if(NULL == device) continue;
    if(NULL != device->device_handle_register)
      if( DAGUE_SUCCESS != device->device_handle_register(device, (dague_handle_t*)__dague_handle) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register handle %p", device->name, __dague_handle);
        continue;
      }
    if(NULL != device->device_memory_register) {  /* Register all the data */
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataA2;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataT1;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataT2;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ1;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataQ2;
      if( (NULL != dague_ddesc->register_memory) &&
          (DAGUE_SUCCESS != dague_ddesc->register_memory(dague_ddesc, device)) ) {
        dague_debug_verbose(3, dague_debug_output, "Device %s refused to register memory for data %s (%p) from handle %p",
                     device->name, dague_ddesc->key_base, dague_ddesc, __dague_handle);
        continue;
      }
    }
    supported_dev |= device->type;
    __dague_handle->super.super.devices_mask |= (1 << _i);
  }
  /* Remove all the chores without a backend device */
  uint32_t i;
  for( i = 0; i < __dague_handle->super.super.nb_functions; i++ ) {
    dague_function_t* func = (dague_function_t*)__dague_handle->super.super.functions_array[i];
    __dague_chore_t* chores = (__dague_chore_t*)func->incarnations;
    uint32_t index = 0;
    uint32_t j;
    for( j = 0; NULL != chores[j].hook; j++ ) {
      if(supported_dev & chores[j].type) {
          if( j != index ) {
            chores[index] = chores[j];
            dague_debug_verbose(20, dague_debug_output, "Device type %i disabled for function %s"
, chores[j].type, func->name);
          }
          index++;
      }
    }
    chores[index].type     = DAGUE_DEV_NONE;
    chores[index].evaluate = NULL;
    chores[index].hook     = NULL;
    dague_execution_context_t* task = (dague_execution_context_t*)dague_thread_mempool_allocate(context->virtual_processes[0]->execution_units[0]->context_mempool);
    task->dague_handle = (dague_handle_t *)__dague_handle;
    task->chore_id = 0;
    task->status = DAGUE_TASK_STATUS_NONE;
    memset(&task->locals, 0, sizeof(assignment_t) * MAX_LOCAL_COUNT);
    DAGUE_LIST_ITEM_SINGLETON(task);
    task->priority = -1;
    task->function = task->dague_handle->functions_array[task->dague_handle->nb_functions + i];
    if( 0 == i ) ready_tasks[0] = &task->super.list_item;
    else ready_tasks[0] = dague_list_item_ring_push(ready_tasks[0], &task->super.list_item);
  }
}
static void dorgqr_split_destructor( __dague_dorgqr_split_internal_handle_t *handle )
{
  uint32_t i;
  for( i = 0; i < (2 * handle->super.super.nb_functions); i++ ) {  /* Extra startup function added at the end */
    dague_function_t* func = (dague_function_t*)handle->super.super.functions_array[i];
    free((void*)func->incarnations);
    free(func);
  }
  free(handle->super.super.functions_array); handle->super.super.functions_array = NULL;
  handle->super.super.nb_functions = 0;

  for(i = 0; i < (uint32_t)handle->super.arenas_size; i++) {
    if( handle->super.arenas[i] != NULL ) {
      dague_arena_destruct(handle->super.arenas[i]);
      free(handle->super.arenas[i]); handle->super.arenas[i] = NULL;
    }
  }
  free( handle->super.arenas ); handle->super.arenas = NULL;
  handle->super.arenas_size = 0;
  /* Destroy the data repositories for this object */
   data_repo_destroy_nothreadsafe(handle->repositories[10]);  /* ungqr_dtsmqr2_in_A2 */
   data_repo_destroy_nothreadsafe(handle->repositories[9]);  /* ungqr_dtsmqr2_in_T2 */
   data_repo_destroy_nothreadsafe(handle->repositories[8]);  /* ungqr_dtsmqr2 */
   data_repo_destroy_nothreadsafe(handle->repositories[7]);  /* ungqr_dtsmqr_in_A1 */
   data_repo_destroy_nothreadsafe(handle->repositories[6]);  /* ungqr_dtsmqr_in_T1 */
   data_repo_destroy_nothreadsafe(handle->repositories[5]);  /* ungqr_dtsmqr1 */
   data_repo_destroy_nothreadsafe(handle->repositories[4]);  /* ungqr_dormqr_in_A1 */
   data_repo_destroy_nothreadsafe(handle->repositories[3]);  /* ungqr_dormqr_in_T1 */
   data_repo_destroy_nothreadsafe(handle->repositories[2]);  /* ungqr_dormqr1 */
   data_repo_destroy_nothreadsafe(handle->repositories[1]);  /* ungqr_dlaset2 */
   data_repo_destroy_nothreadsafe(handle->repositories[0]);  /* ungqr_dlaset1 */
  /* Release the dependencies arrays for this object */
  dague_destruct_dependencies( handle->super.super.dependencies_array[10] );
  handle->super.super.dependencies_array[10] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[9] );
  handle->super.super.dependencies_array[9] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[8] );
  handle->super.super.dependencies_array[8] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[7] );
  handle->super.super.dependencies_array[7] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[6] );
  handle->super.super.dependencies_array[6] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[5] );
  handle->super.super.dependencies_array[5] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[4] );
  handle->super.super.dependencies_array[4] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[3] );
  handle->super.super.dependencies_array[3] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[2] );
  handle->super.super.dependencies_array[2] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[1] );
  handle->super.super.dependencies_array[1] = NULL;
  dague_destruct_dependencies( handle->super.super.dependencies_array[0] );
  handle->super.super.dependencies_array[0] = NULL;
  free( handle->super.super.dependencies_array );
  handle->super.super.dependencies_array = NULL;
  /* Unregister all the data */
  uint32_t _i;
  for( _i = 0; _i < dague_nb_devices; _i++ ) {
    dague_device_t* device;
    dague_ddesc_t* dague_ddesc;
    if(!(handle->super.super.devices_mask & (1 << _i))) continue;
    if((NULL == (device = dague_devices_get(_i))) || (NULL == device->device_memory_unregister)) continue;
    dague_ddesc = (dague_ddesc_t*)handle->super.dataA1;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
  dague_ddesc = (dague_ddesc_t*)handle->super.dataA2;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
  dague_ddesc = (dague_ddesc_t*)handle->super.dataT1;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
  dague_ddesc = (dague_ddesc_t*)handle->super.dataT2;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
  dague_ddesc = (dague_ddesc_t*)handle->super.dataQ1;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
  dague_ddesc = (dague_ddesc_t*)handle->super.dataQ2;
  if( NULL != dague_ddesc->unregister_memory ) { (void)dague_ddesc->unregister_memory(dague_ddesc, device); };
}
  /* Unregister the handle from the devices */
  for( i = 0; i < dague_nb_devices; i++ ) {
    if(!(handle->super.super.devices_mask & (1 << i))) continue;
    handle->super.super.devices_mask ^= (1 << i);
    dague_device_t* device = dague_devices_get(i);
    if((NULL == device) || (NULL == device->device_handle_unregister)) continue;
    if( DAGUE_SUCCESS != device->device_handle_unregister(device, &handle->super.super) ) continue;
  }
  dague_handle_unregister( &handle->super.super );
  free(handle);
}

#undef dataA1
#undef dataA2
#undef dataT1
#undef dataT2
#undef dataQ1
#undef dataQ2
#undef optid
#undef p_work
#undef descA1
#undef descA2
#undef descT1
#undef descT2
#undef descQ1
#undef descQ2
#undef ib
#undef KT
#undef KT2

dague_dorgqr_split_handle_t *dague_dorgqr_split_new(dague_ddesc_t * dataA1 /* data dataA1 */, dague_ddesc_t * dataA2 /* data dataA2 */, dague_ddesc_t * dataT1 /* data dataT1 */, dague_ddesc_t * dataT2 /* data dataT2 */, dague_ddesc_t * dataQ1 /* data dataQ1 */, dague_ddesc_t * dataQ2 /* data dataQ2 */, int optid, dague_memory_pool_t * p_work)
{
  __dague_dorgqr_split_internal_handle_t *__dague_handle = (__dague_dorgqr_split_internal_handle_t *)calloc(1, sizeof(__dague_dorgqr_split_internal_handle_t));
  dague_function_t* func;
  int i, j;
  /* Dump the hidden parameters */
  tiled_matrix_desc_t descA1;
  tiled_matrix_desc_t descA2;
  tiled_matrix_desc_t descT1;
  tiled_matrix_desc_t descT2;
  tiled_matrix_desc_t descQ1;
  tiled_matrix_desc_t descQ2;
  int ib;
  int KT;
  int KT2;
  __dague_handle->super.super.nb_functions = DAGUE_dorgqr_split_NB_FUNCTIONS;
  __dague_handle->super.super.devices_mask = DAGUE_DEVICES_ALL;
  __dague_handle->super.super.dependencies_array = (void **)
              calloc(__dague_handle->super.super.nb_functions , sizeof(void*));
  /* Twice the size to hold the startup tasks function_t */
  __dague_handle->super.super.functions_array = (const dague_function_t**)
              malloc(2 * __dague_handle->super.super.nb_functions * sizeof(dague_function_t*));
  __dague_handle->super.super.nb_tasks = __dague_handle->super.super.nb_functions;
  __dague_handle->super.super.nb_pending_actions = 1;  /* for the startup tasks */
  __dague_handle->sync_point = __dague_handle->super.super.nb_functions;
  __dague_handle->startup_queue = NULL;
  for( i = 0; i < (int)__dague_handle->super.super.nb_functions; i++ ) {
    __dague_handle->super.super.functions_array[i] = func = malloc(sizeof(dague_function_t));
    memcpy((dague_function_t*)__dague_handle->super.super.functions_array[i], dorgqr_split_functions[i], sizeof(dague_function_t));
    for( j = 0; NULL != func->incarnations[j].hook; j++);
    func->incarnations = (__dague_chore_t*)malloc((j+1) * sizeof(__dague_chore_t));
    memcpy((__dague_chore_t*)func->incarnations, dorgqr_split_functions[i]->incarnations, (j+1) * sizeof(__dague_chore_t));

    /* Add a placeholder for initialization and startup task */
    __dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+i] = func = (dague_function_t*)malloc(sizeof(dague_function_t));
    memcpy(func, (void*)&__dague_generic_startup, sizeof(dague_function_t));
    func->function_id = __dague_handle->super.super.nb_functions + i;
    func->incarnations = (__dague_chore_t*)malloc(2 * sizeof(__dague_chore_t));
    memcpy((__dague_chore_t*)func->incarnations, (void*)__dague_generic_startup.incarnations, 2 * sizeof(__dague_chore_t));
  }
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+0];
  func->name = "Generic Startup for ungqr_dtsmqr2_in_A2";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr2_in_A2_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dtsmqr2_in_A2;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+1];
  func->name = "Generic Startup for ungqr_dtsmqr2_in_T2";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr2_in_T2_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dtsmqr2_in_T2;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+2];
  func->name = "Generic Startup for ungqr_dtsmqr2";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr2_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+3];
  func->name = "Generic Startup for ungqr_dtsmqr_in_A1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr_in_A1_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dtsmqr_in_A1;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+4];
  func->name = "Generic Startup for ungqr_dtsmqr_in_T1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr_in_T1_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dtsmqr_in_T1;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+5];
  func->name = "Generic Startup for ungqr_dtsmqr1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dtsmqr1_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+6];
  func->name = "Generic Startup for ungqr_dormqr_in_A1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dormqr_in_A1_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dormqr_in_A1;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+7];
  func->name = "Generic Startup for ungqr_dormqr_in_T1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dormqr_in_T1_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dormqr_in_T1;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+8];
  func->name = "Generic Startup for ungqr_dormqr1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dormqr1_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+9];
  func->name = "Generic Startup for ungqr_dlaset2";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dlaset2_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dlaset2;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+10];
  func->name = "Generic Startup for ungqr_dlaset1";
  func->prepare_input = (dague_hook_t*)dorgqr_split_ungqr_dlaset1_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_ungqr_dlaset1;
  /* Compute the number of arenas: */
  /*   DAGUE_dorgqr_split_DEFAULT_ARENA  ->  0 */
  /*   DAGUE_dorgqr_split_LOWER_TILE_ARENA  ->  1 */
  /*   DAGUE_dorgqr_split_LITTLE_T_ARENA  ->  2 */
  __dague_handle->super.arenas_size = 3;
  __dague_handle->super.arenas = (dague_arena_t **)malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t*));
  for(i = 0; i < __dague_handle->super.arenas_size; i++) {
    __dague_handle->super.arenas[i] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));
  }
  /* Now the Parameter-dependent structures: */
  __dague_handle->super.dataA1 = dataA1;
  __dague_handle->super.dataA2 = dataA2;
  __dague_handle->super.dataT1 = dataT1;
  __dague_handle->super.dataT2 = dataT2;
  __dague_handle->super.dataQ1 = dataQ1;
  __dague_handle->super.dataQ2 = dataQ2;
  __dague_handle->super.optid = optid;
  __dague_handle->super.p_work = p_work;
  __dague_handle->super.descA1 = descA1 = *((tiled_matrix_desc_t*)dataA1);
  __dague_handle->super.descA2 = descA2 = *((tiled_matrix_desc_t*)dataA2);
  __dague_handle->super.descT1 = descT1 = *((tiled_matrix_desc_t*)dataT1);
  __dague_handle->super.descT2 = descT2 = *((tiled_matrix_desc_t*)dataT2);
  __dague_handle->super.descQ1 = descQ1 = *((tiled_matrix_desc_t*)dataQ1);
  __dague_handle->super.descQ2 = descQ2 = *((tiled_matrix_desc_t*)dataQ2);
  __dague_handle->super.ib = ib = descT1.mb;
  __dague_handle->super.KT = KT = descA1.nt-1;
  __dague_handle->super.KT2 = KT2 = dague_imin( KT, descQ1.mt-2 );
  /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
  __dague_handle->super.super.profiling_array = dorgqr_split_profiling_array;
  if( -1 == dorgqr_split_profiling_array[0] ) {
    dague_profiling_add_dictionary_keyword("ungqr_dtsmqr2", "fill:CC2828",
                                       sizeof(dague_profile_ddesc_info_t)+4*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};mmax{int32_t};m{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dorgqr_split_ungqr_dtsmqr2.function_id /* ungqr_dtsmqr2 start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dorgqr_split_ungqr_dtsmqr2.function_id /* ungqr_dtsmqr2 end key */]);

    dague_profiling_add_dictionary_keyword("ungqr_dtsmqr1", "fill:CC8128",
                                       sizeof(dague_profile_ddesc_info_t)+3*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};m{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dorgqr_split_ungqr_dtsmqr1.function_id /* ungqr_dtsmqr1 start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dorgqr_split_ungqr_dtsmqr1.function_id /* ungqr_dtsmqr1 end key */]);

    dague_profiling_add_dictionary_keyword("ungqr_dormqr1", "fill:BDCC28",
                                       sizeof(dague_profile_ddesc_info_t)+2*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dorgqr_split_ungqr_dormqr1.function_id /* ungqr_dormqr1 start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dorgqr_split_ungqr_dormqr1.function_id /* ungqr_dormqr1 end key */]);

  }
#  endif /* defined(DAGUE_PROF_TRACE) */
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2_in_A2);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2_in_T2);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr2);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr_in_A1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr_in_T1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dtsmqr1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr_in_A1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr_in_T1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dormqr1);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dlaset2);
  AYU_REGISTER_TASK(&dorgqr_split_ungqr_dlaset1);
  __dague_handle->super.super.repo_array = __dague_handle->repositories;
  __dague_handle->super.super.startup_hook = (dague_startup_fn_t)dorgqr_split_startup;
  __dague_handle->super.super.destructor   = (dague_destruct_fn_t)dorgqr_split_destructor;
  (void)dague_handle_reserve_id((dague_handle_t*)__dague_handle);
  return (dague_dorgqr_split_handle_t*)__dague_handle;
}

