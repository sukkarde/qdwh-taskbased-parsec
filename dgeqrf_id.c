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

#define DAGUE_dgeqrf_id_NB_FUNCTIONS 6
#define DAGUE_dgeqrf_id_NB_DATA 3

typedef struct __dague_dgeqrf_id_internal_handle_s __dague_dgeqrf_id_internal_handle_t;
struct dague_dgeqrf_id_internal_handle_s;

/** Predeclarations of the dague_function_t */
static const dague_function_t dgeqrf_id_geqrf_dtsmqr;
static const dague_function_t dgeqrf_id_geqrf_dtsmqr_out_A1;
static const dague_function_t dgeqrf_id_geqrf_dtsqrt;
static const dague_function_t dgeqrf_id_geqrf_dtsqrt_out_A1;
static const dague_function_t dgeqrf_id_A2_in;
static const dague_function_t dgeqrf_id_A1_in;
/** Predeclarations of the parameters */
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_V;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_T;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_T;
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1;
static const dague_flow_t flow_of_dgeqrf_id_A2_in_for_A;
static const dague_flow_t flow_of_dgeqrf_id_A1_in_for_A;
#line 2 "dgeqrf_id.jdf"
/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 *
 * @generated d Sat Aug 13 16:38:52 2016
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


#line 72 "dgeqrf_id.c"
#include "dgeqrf_id.h"

struct __dague_dgeqrf_id_internal_handle_s {
 dague_dgeqrf_id_handle_t super;
 volatile uint32_t sync_point;
 dague_execution_context_t* startup_queue;
  /* The ranges to compute the hash key */
  int geqrf_dtsmqr_k_range;
  int geqrf_dtsmqr_m_range;
  int geqrf_dtsmqr_n_range;
  int geqrf_dtsmqr_out_A1_k_range;
  int geqrf_dtsmqr_out_A1_n_range;
  int geqrf_dtsqrt_k_range;
  int geqrf_dtsqrt_m_range;
  int geqrf_dtsqrt_out_A1_k_range;
  int A2_in_m_range;
  int A2_in_n_range;
  int A1_in_m_range;
  int A1_in_n_range;
  /* The list of data repositories  geqrf_dtsmqr  geqrf_dtsmqr_out_A1  geqrf_dtsqrt  geqrf_dtsqrt_out_A1  A2_in  A1_in */
  data_repo_t* repositories[6];
};

#if defined(DAGUE_PROF_TRACE)
static int dgeqrf_id_profiling_array[2*DAGUE_dgeqrf_id_NB_FUNCTIONS] = {-1};
#endif  /* defined(DAGUE_PROF_TRACE) */
/* Globals */
#define ib (__dague_handle->super.ib)
#define optid (__dague_handle->super.optid)
#define p_work (__dague_handle->super.p_work)
#define p_tau (__dague_handle->super.p_tau)
#define descA1 (__dague_handle->super.descA1)
#define descA2 (__dague_handle->super.descA2)
#define descT2 (__dague_handle->super.descT2)
#define KT (__dague_handle->super.KT)
#define smallnb (__dague_handle->super.smallnb)

/* Data Access Macros */
#define dataT2(dataT20,dataT21)  (((dague_ddesc_t*)__dague_handle->super.dataT2)->data_of((dague_ddesc_t*)__dague_handle->super.dataT2, (dataT20), (dataT21)))

#define dataA2(dataA20,dataA21)  (((dague_ddesc_t*)__dague_handle->super.dataA2)->data_of((dague_ddesc_t*)__dague_handle->super.dataA2, (dataA20), (dataA21)))

#define dataA1(dataA10,dataA11)  (((dague_ddesc_t*)__dague_handle->super.dataA1)->data_of((dague_ddesc_t*)__dague_handle->super.dataA1, (dataA10), (dataA11)))


/* Functions Predicates */
#define geqrf_dtsmqr_pred(k, mmax, m, n) (((dague_ddesc_t*)(__dague_handle->super.dataA2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n))
#define geqrf_dtsmqr_out_A1_pred(k, n, mmax) (((dague_ddesc_t*)(__dague_handle->super.dataA1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, n))
#define geqrf_dtsqrt_pred(k, mmax, m) (((dague_ddesc_t*)(__dague_handle->super.dataA2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, k))
#define geqrf_dtsqrt_out_A1_pred(k, mmax) (((dague_ddesc_t*)(__dague_handle->super.dataA1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, k))
#define A2_in_pred(m, nmin, n) (((dague_ddesc_t*)(__dague_handle->super.dataA2))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA2))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n))
#define A1_in_pred(m, n) (((dague_ddesc_t*)(__dague_handle->super.dataA1))->myrank == ((dague_ddesc_t*)(__dague_handle->super.dataA1))->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, n))

/* Data Repositories */
#define geqrf_dtsmqr_repo (__dague_handle->repositories[5])
#define geqrf_dtsqrt_repo (__dague_handle->repositories[3])
#define A2_in_repo (__dague_handle->repositories[1])
#define A1_in_repo (__dague_handle->repositories[0])
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
static inline int dgeqrf_id_geqrf_dtsmqr_inline_c_expr1_line_255(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsmqr */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  (void)k;  (void)mmax;  (void)m;  (void)n;

 return n; 
#line 184 "dgeqrf_id.c"
}

static inline int dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsmqr */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  (void)k;  (void)mmax;  (void)m;  (void)n;

 return optid ? k : descA2.mt-1; 
#line 199 "dgeqrf_id.c"
}

static inline int dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsmqr */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  (void)k;  (void)mmax;  (void)m;  (void)n;

 return dague_imin(KT,(descA1.nt-2)); 
#line 214 "dgeqrf_id.c"
}

static inline int dgeqrf_id_geqrf_dtsmqr_out_A1_inline_c_expr4_line_215(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsmqr_out_A1 */
  const int k = assignments->k.value;
  const int n = assignments->n.value;
  const int mmax = assignments->mmax.value;

  (void)k;  (void)n;  (void)mmax;

 return optid ? k : descA2.mt-1; 
#line 228 "dgeqrf_id.c"
}

static inline int dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsqrt */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;
  const int m = assignments->m.value;

  (void)k;  (void)mmax;  (void)m;

 return optid ? k : descA2.mt-1; 
#line 242 "dgeqrf_id.c"
}

static inline int dgeqrf_id_geqrf_dtsqrt_out_A1_inline_c_expr6_line_92(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *assignments)
{
  (void)__dague_handle;
  /* This inline C function was declared in the context of the task geqrf_dtsqrt_out_A1 */
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;

  (void)k;  (void)mmax;

 return optid ? k : descA2.mt-1; 
#line 255 "dgeqrf_id.c"
}

static inline uint64_t __jdf2c_hash_geqrf_dtsmqr(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = (k + 1);
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->geqrf_dtsmqr_k_range;
  __h += (n - __jdf2c_n_min) * __dague_handle->geqrf_dtsmqr_k_range * __dague_handle->geqrf_dtsmqr_m_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_geqrf_dtsmqr_out_A1(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = (k + 1);
  const int mmax = assignments->mmax.value;
  (void)mmax;
  __h += (k - __jdf2c_k_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->geqrf_dtsmqr_out_A1_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_geqrf_dtsqrt(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  __h += (k - __jdf2c_k_min);
  __h += (m - __jdf2c_m_min) * __dague_handle->geqrf_dtsqrt_k_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_geqrf_dtsqrt_out_A1(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int k = assignments->k.value;
  int __jdf2c_k_min = 0;
  const int mmax = assignments->mmax.value;
  (void)mmax;
  __h += (k - __jdf2c_k_min);
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_A2_in(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_A2_in_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int nmin = assignments->nmin.value;
  (void)nmin;
  const int n = assignments->n.value;
  int __jdf2c_n_min = nmin;
  __h += (m - __jdf2c_m_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->A2_in_m_range;
 (void)__dague_handle; return __h;
}

static inline uint64_t __jdf2c_hash_A1_in(const __dague_dgeqrf_id_internal_handle_t *__dague_handle,
                          const __dague_dgeqrf_id_A1_in_assignment_t *assignments)
{
  uint64_t __h = 0;
  const int m = assignments->m.value;
  int __jdf2c_m_min = 0;
  const int n = assignments->n.value;
  int __jdf2c_n_min = m;
  __h += (m - __jdf2c_m_min);
  __h += (n - __jdf2c_n_min) * __dague_handle->A1_in_m_range;
 (void)__dague_handle; return __h;
}

static inline int priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *assignments);
static inline int priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments);
/******                                  geqrf_dtsmqr                                  ******/

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, locals);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_k, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dgeqrf_id_geqrf_dtsmqr_mmax_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, locals);
}
static const expr_t expr_of_symb_dgeqrf_id_geqrf_dtsmqr_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dgeqrf_id_geqrf_dtsmqr_mmax_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dgeqrf_id_geqrf_dtsmqr_mmax, .max = &expr_of_symb_dgeqrf_id_geqrf_dtsmqr_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return mmax;
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_m = { .name = "m", .context_index = 2, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k + 1);
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA1.nt - 1);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_n = { .name = "n", .context_index = 3, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_geqrf_dtsmqr(__dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (((descA1.mt - k) * (descA1.mt - n)) * (descA1.mt - n));
}
static const expr_t priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct }
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == 0);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue,  /* (m == 0) */
  .ctl_gather_nb = NULL,
  .function_id = 0, /* dgeqrf_id_A1_in */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_A1_in_for_A,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return !(m == 0);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse,  /* !(m == 0) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == mmax);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue,  /* (m == mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 4, /* dgeqrf_id_geqrf_dtsmqr_out_A1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return !(m == mmax);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse,  /* !(m == mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue,
 &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue,
 &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse }
};

static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return ((!optid && (k == 0)) || (optid && (k == m)));
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243,  /* ((!optid && (k == 0)) || (optid && (k == m))) */
  .ctl_gather_nb = NULL,
  .function_id = 1, /* dgeqrf_id_A2_in */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_A2_in_for_A,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep2_atline_244 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
  .dep_index = 2,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return ((k + 1) == n);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246,  /* ((k + 1) == n) */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *locals)
{
  const int k = locals->k.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return ((k + 1) < n);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247,  /* ((k + 1) < n) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
  .dep_index = 2,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2 = {
  .name               = "A2",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 1,
  .flow_datatype_mask = 0x2,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243,
 &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep2_atline_244 },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246,
 &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247 }
};

static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_V_dep1_atline_249 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_V = {
  .name               = "V",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 2,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V_dep1_atline_249 },
  .dep_out    = { NULL }
};

static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_T_dep1_atline_250 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_for_T = {
  .name               = "T",
  .sym_type           = SYM_IN,
  .flow_flags         = FLOW_ACCESS_READ,
  .flow_index         = 3,
  .flow_datatype_mask = 0x0,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T_dep1_atline_250 },
  .dep_out    = { NULL }
};

static void
iterate_successors_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
      if( (m == mmax) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr_out_A1.function_id];
        const int geqrf_dtsmqr_out_A1_k = k;
        if( (geqrf_dtsmqr_out_A1_k >= (0)) && (geqrf_dtsmqr_out_A1_k <= ((descA1.nt - 2))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_out_A1_k;
          const int geqrf_dtsmqr_out_A1_n = n;
          if( (geqrf_dtsmqr_out_A1_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_out_A1_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = geqrf_dtsmqr_out_A1_n;
            const int geqrf_dtsmqr_out_A1_mmax = dgeqrf_id_geqrf_dtsmqr_out_A1_inline_c_expr4_line_215(__dague_handle, &ncc->locals);
            assert(&nc.locals[2].value == &ncc->locals.mmax.value);
            ncc->locals.mmax.value = geqrf_dtsmqr_out_A1_mmax;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iftrue, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    } else {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = k;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = (m + 1);
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            const int geqrf_dtsmqr_n = n;
            if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep2_atline_241_iffalse, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  if( action_mask & 0x6 ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x2 ) {
        if( ((k + 1) == n) ) {
      __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
        const int geqrf_dtsqrt_k = (k + 1);
        if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_k;
          const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
          const int geqrf_dtsqrt_m = m;
          if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep3_atline_246, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x4 ) {
        if( ((k + 1) < n) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = (k + 1);
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = m;
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            const int geqrf_dtsmqr_n = n;
            if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
            RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep4_atline_247, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
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
iterate_predecessors_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == 0) ) {
      __dague_dgeqrf_id_A1_in_task_t* ncc = (__dague_dgeqrf_id_A1_in_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_A1_in.function_id];
        const int A1_in_m = k;
        if( (A1_in_m >= (0)) && (A1_in_m <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = A1_in_m;
          const int A1_in_n = n;
          if( (A1_in_n >= (ncc->locals.m.value)) && (A1_in_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = A1_in_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iftrue, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    } else {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = k;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = (m - 1);
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            const int geqrf_dtsmqr_n = n;
            if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
            RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1_dep1_atline_240_iffalse, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  if( action_mask & 0x6 ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x2 ) {
        if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
      __dague_dgeqrf_id_A2_in_task_t* ncc = (__dague_dgeqrf_id_A2_in_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_A2_in.function_id];
        const int A2_in_m = m;
        if( (A2_in_m >= (0)) && (A2_in_m <= ((descA2.mt - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = A2_in_m;
          const int A2_in_nmin = (optid ? ncc->locals.m.value : 0);
          assert(&nc.locals[1].value == &ncc->locals.nmin.value);
          ncc->locals.nmin.value = A2_in_nmin;
          const int A2_in_n = n;
          if( (A2_in_n >= (ncc->locals.nmin.value)) && (A2_in_n <= ((descA2.nt - 1))) ) {
            assert(&nc.locals[2].value == &ncc->locals.n.value);
            ncc->locals.n.value = A2_in_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep1_atline_243, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x4 ) {
        __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
      const int geqrf_dtsmqr_k = (k - 1);
      if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsmqr_k;
        const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
        const int geqrf_dtsmqr_m = m;
        if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsmqr_m;
          const int geqrf_dtsmqr_n = n;
          if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[3].value == &ncc->locals.n.value);
            ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2_dep2_atline_244, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
        }
  }
  }
  if( action_mask & 0x8 ) {  /* Flow of Data V */
    data.data   = this_task->data.V.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x8 ) {
        __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
      const int geqrf_dtsqrt_k = k;
      if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsqrt_k;
        const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
        const int geqrf_dtsqrt_m = m;
        if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
        RELEASE_DEP_OUTPUT(eu, "V", this_task, "A2", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V_dep1_atline_249, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  if( action_mask & 0x10 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x10 ) {
        __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
      const int geqrf_dtsqrt_k = k;
      if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsqrt_k;
        const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
        const int geqrf_dtsqrt_m = m;
        if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
        RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T_dep1_atline_250, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *eu, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
    arg.output_entry = data_repo_lookup_entry_and_create( eu, geqrf_dtsmqr_repo, __jdf2c_hash_geqrf_dtsmqr(__dague_handle, (__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dgeqrf_id_geqrf_dtsmqr(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(geqrf_dtsmqr_repo, arg.output_entry->key, arg.output_usage);
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

    if( (m == 0) ) {
      data_repo_entry_used_once( eu, A1_in_repo, this_task->data.A1.data_repo->key );
    } else {
      data_repo_entry_used_once( eu, geqrf_dtsmqr_repo, this_task->data.A1.data_repo->key );
    }
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
    if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
      data_repo_entry_used_once( eu, A2_in_repo, this_task->data.A2.data_repo->key );
    }
    else {
    data_repo_entry_used_once( eu, geqrf_dtsmqr_repo, this_task->data.A2.data_repo->key );
    }
    if( NULL != this_task->data.A2.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A2.data_in);
    }
    data_repo_entry_used_once( eu, geqrf_dtsqrt_repo, this_task->data.V.data_repo->key );
    if( NULL != this_task->data.V.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.V.data_in);
    }
    data_repo_entry_used_once( eu, geqrf_dtsqrt_repo, this_task->data.T.data_repo->key );
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
    if( (m == 0) ) {
__dague_dgeqrf_id_A1_in_assignment_t *target_locals = (__dague_dgeqrf_id_A1_in_assignment_t*)&generic_locals;
      const int A1_inm = target_locals->m.value = k; (void)A1_inm;
      const int A1_inn = target_locals->n.value = n; (void)A1_inn;
      entry = data_repo_lookup_entry( A1_in_repo, __jdf2c_hash_A1_in( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:A1_in to A1:geqrf_dtsmqr");
      }
      chunk = entry->data[0];  /* A1:geqrf_dtsmqr <- A:A1_in */
      ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_A1_in, "A", target_locals, chunk);
    } else {
__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t*)&generic_locals;
      const int geqrf_dtsmqrk = target_locals->k.value = k; (void)geqrf_dtsmqrk;
      const int geqrf_dtsmqrmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, target_locals); (void)geqrf_dtsmqrmmax;
      const int geqrf_dtsmqrm = target_locals->m.value = (m - 1); (void)geqrf_dtsmqrm;
      const int geqrf_dtsmqrn = target_locals->n.value = n; (void)geqrf_dtsmqrn;
      entry = data_repo_lookup_entry( geqrf_dtsmqr_repo, __jdf2c_hash_geqrf_dtsmqr( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:geqrf_dtsmqr to A1:geqrf_dtsmqr");
      }
      chunk = entry->data[0];  /* A1:geqrf_dtsmqr <- A1:geqrf_dtsmqr */
      ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_geqrf_dtsmqr, "A1", target_locals, chunk);
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
    if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
__dague_dgeqrf_id_A2_in_assignment_t *target_locals = (__dague_dgeqrf_id_A2_in_assignment_t*)&generic_locals;
      const int A2_inm = target_locals->m.value = m; (void)A2_inm;
      const int A2_innmin = target_locals->nmin.value = (optid ? A2_inm : 0); (void)A2_innmin;
      const int A2_inn = target_locals->n.value = n; (void)A2_inn;
      entry = data_repo_lookup_entry( A2_in_repo, __jdf2c_hash_A2_in( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:A2_in to A2:geqrf_dtsmqr");
      }
      chunk = entry->data[0];  /* A2:geqrf_dtsmqr <- A:A2_in */
      ACQUIRE_FLOW(this_task, "A2", &dgeqrf_id_A2_in, "A", target_locals, chunk);
    }
    else {
__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t*)&generic_locals;
      const int geqrf_dtsmqrk = target_locals->k.value = (k - 1); (void)geqrf_dtsmqrk;
      const int geqrf_dtsmqrmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, target_locals); (void)geqrf_dtsmqrmmax;
      const int geqrf_dtsmqrm = target_locals->m.value = m; (void)geqrf_dtsmqrm;
      const int geqrf_dtsmqrn = target_locals->n.value = n; (void)geqrf_dtsmqrn;
      entry = data_repo_lookup_entry( geqrf_dtsmqr_repo, __jdf2c_hash_geqrf_dtsmqr( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A2:geqrf_dtsmqr to A2:geqrf_dtsmqr");
      }
      chunk = entry->data[1];  /* A2:geqrf_dtsmqr <- A2:geqrf_dtsmqr */
      ACQUIRE_FLOW(this_task, "A2", &dgeqrf_id_geqrf_dtsmqr, "A2", target_locals, chunk);
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
__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t*)&generic_locals;
  const int geqrf_dtsqrtk = target_locals->k.value = k; (void)geqrf_dtsqrtk;
  const int geqrf_dtsqrtmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, target_locals); (void)geqrf_dtsqrtmmax;
  const int geqrf_dtsqrtm = target_locals->m.value = m; (void)geqrf_dtsqrtm;
    entry = data_repo_lookup_entry( geqrf_dtsqrt_repo, __jdf2c_hash_geqrf_dtsqrt( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from A2:geqrf_dtsqrt to V:geqrf_dtsmqr");
    }
    chunk = entry->data[1];  /* V:geqrf_dtsmqr <- A2:geqrf_dtsqrt */
    ACQUIRE_FLOW(this_task, "V", &dgeqrf_id_geqrf_dtsqrt, "A2", target_locals, chunk);
      this_task->data.V.data_in   = chunk;   /* flow V */
      this_task->data.V.data_repo = entry;
    }
    this_task->data.V.data_out = NULL;  /* input only */

  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t*)&generic_locals;
  const int geqrf_dtsqrtk = target_locals->k.value = k; (void)geqrf_dtsqrtk;
  const int geqrf_dtsqrtmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, target_locals); (void)geqrf_dtsqrtmmax;
  const int geqrf_dtsqrtm = target_locals->m.value = m; (void)geqrf_dtsqrtm;
    entry = data_repo_lookup_entry( geqrf_dtsqrt_repo, __jdf2c_hash_geqrf_dtsqrt( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from T:geqrf_dtsqrt to T:geqrf_dtsmqr");
    }
    chunk = entry->data[2];  /* T:geqrf_dtsmqr <- T:geqrf_dtsqrt */
    ACQUIRE_FLOW(this_task, "T", &dgeqrf_id_geqrf_dtsqrt, "T", target_locals, chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    this_task->data.T.data_out = NULL;  /* input only */

  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataA2;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataA2))->data_key((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)k;  (void)mmax;  (void)m;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x2U) && (((!optid && (k == 0)) || (optid && (k == m)))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x2U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow V */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x4U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x4U) */
if( (*flow_mask) & 0x8U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x8U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x8U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x6U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x6U) && (((k + 1) == n) || ((k + 1) < n)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x6U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x6U) */
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
#if defined(DAGUE_HAVE_CUDA)
struct dague_body_cuda_dgeqrf_id_geqrf_dtsmqr_s {
  uint8_t      index;
  cudaStream_t stream;
  void*           dyld_fn;
};

static int gpu_kernel_submit_dgeqrf_id_geqrf_dtsmqr(gpu_device_t            *gpu_device,
                                   dague_gpu_context_t     *gpu_task,
                                   dague_gpu_exec_stream_t *gpu_stream )
{
  __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t *)gpu_task->ec;
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  struct dague_body_cuda_dgeqrf_id_geqrf_dtsmqr_s dague_body = { gpu_device->cuda_index, gpu_stream->cuda_stream, NULL };
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)k;  (void)mmax;  (void)m;  (void)n;

  (void)gpu_device; (void)gpu_stream; (void)__dague_handle; (void)dague_body;
  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_out;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;
  dague_data_copy_t *gA2 = this_task->data.A2.data_out;
  void *A2 = (NULL != gA2) ? DAGUE_DATA_COPY_GET_PTR(gA2) : NULL; (void)A2;
  dague_data_copy_t *gV = this_task->data.V.data_out;
  void *V = (NULL != gV) ? DAGUE_DATA_COPY_GET_PTR(gV) : NULL; (void)V;
  dague_data_copy_t *gT = this_task->data.T.data_out;
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
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
  cache_buf_referenced(context->closest_cache, A2);
  cache_buf_referenced(context->closest_cache, V);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */
#if defined(DAGUE_DEBUG_NOISIER)
  {
    char tmp[MAX_TASK_STRLEN];
    DAGUE_DEBUG_VERBOSE(10, dague_cuda_output_stream, "GPU[%1d]:\tEnqueue on device %s priority %d\n", gpu_device->cuda_index, 
           dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (dague_execution_context_t *)this_task),
           this_task->priority );
  }
#endif /* defined(DAGUE_DEBUG_NOISIER) */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              geqrf_dtsmqr BODY                                -----*/

  DAGUE_TASK_PROF_TRACE_IF(gpu_stream->prof_event_track_enable,
                           gpu_stream->profiling,
                           (-1 == gpu_stream->prof_event_key_start ?
                           DAGUE_PROF_FUNC_KEY_START(this_task->dague_handle,
                                                     this_task->function->function_id) :
                           gpu_stream->prof_event_key_start),
                           this_task);
#line 256 "dgeqrf_id.jdf"
{
    double *WORK, *WORKC;
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m- m * descA2.mb) : descA2.mb;
    int tempnn = (n == (descA2.nt-1)) ? (descA2.n- n * descA2.nb) : descA2.nb;
    int ldak = BLKLDD( descA1, k );
    int ldam = BLKLDD( descA2, m );

    WORK  = dague_gpu_pop_workspace(gpu_device, gpu_stream, descA2.nb * ib * sizeof(double));
    WORKC = dague_gpu_pop_workspace(gpu_device, gpu_stream, descA2.mb * ib * sizeof(double));

    dplasma_cuda_dtsmqr( PlasmaLeft, PlasmaTrans,
                         descA1.mb, tempnn, tempmm, tempnn, descA2.nb, ib,
                         A1 /* dataA1(k,n) */, ldak,
                         A2 /* dataA2(m,n) */, ldam,
                         V  /* dataA2(m,k) */, ldam,
                         T  /* dataT2(m,k) */, descT2.mb,
                         WORK,  ib,
                         WORKC, descA2.mb,
                         dague_body.stream );
}

#line 1440 "dgeqrf_id.c"
/*-----                            END OF geqrf_dtsmqr BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}

static int hook_of_dgeqrf_id_geqrf_dtsmqr_CUDA(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  dague_gpu_context_t *gpu_task;
  double ratio;
  int dev_index;
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)k;  (void)mmax;  (void)m;  (void)n;

  (void)context; (void)__dague_handle;

  ratio = 1.;
  dev_index = dgeqrf_id_geqrf_dtsmqr_inline_c_expr1_line_255(__dague_handle, &this_task->locals);
  if (dev_index < -1) {
    return DAGUE_HOOK_RETURN_NEXT;
  } else if (dev_index == -1) {
    dev_index = dague_gpu_get_best_device((dague_execution_context_t*)this_task, ratio);
  } else {
    dev_index = (dev_index % (dague_devices_enabled()-2)) + 2;
  }
  assert(dev_index >= 0);
  if( dev_index < 2 ) {
    return DAGUE_HOOK_RETURN_NEXT;  /* Fall back */
  }
  dague_device_load[dev_index] += ratio * dague_device_sweight[dev_index];

  gpu_task = (dague_gpu_context_t*)calloc(1, sizeof(dague_gpu_context_t));
  OBJ_CONSTRUCT(gpu_task, dague_list_item_t);
  gpu_task->ec = (dague_execution_context_t*)this_task;
  gpu_task->submit = &gpu_kernel_submit_dgeqrf_id_geqrf_dtsmqr;
  gpu_task->task_type = 0;
  gpu_task->pushout[0] = 0;
  gpu_task->flow[0]    = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1;
  if( (m == mmax) ) {
    gpu_task->pushout[0] = 1;
  }
  gpu_task->pushout[1] = 0;
  gpu_task->flow[1]    = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2;
  if( ((k + 1) == n) ) {
    gpu_task->pushout[1] = 1;
  }  gpu_task->pushout[2] = 0;
  gpu_task->flow[2]    = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V;
  gpu_task->pushout[3] = 0;
  gpu_task->flow[3]    = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T;

  return dague_gpu_kernel_scheduler( context, gpu_task, dev_index );
}

#endif  /*  defined(DAGUE_HAVE_CUDA) */
#if defined(DAGUE_HAVE_RECURSIVE)
static int hook_of_dgeqrf_id_geqrf_dtsmqr_RECURSIVE(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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

/*-----                              geqrf_dtsmqr BODY                                -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 279 "dgeqrf_id.jdf"
{
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m- m * descA2.mb) : descA2.mb;
    int tempnn = (n == (descA2.nt-1)) ? (descA2.n- n * descA2.nb) : descA2.nb;
    int ldak = BLKLDD( descA1, k );
    int ldam = BLKLDD( descA2, m );

    if (tempmm > smallnb) {
        subtile_desc_t *small_descA1;
        subtile_desc_t *small_descA2;
        subtile_desc_t *small_descV;
        subtile_desc_t *small_descT;
        dague_handle_t *dague_dtsmqr;


        small_descA1 = subtile_desc_create( &(descA1), k, n,
                                            dague_imin(descA1.mb, ldak), smallnb,
                                            0, 0, descA1.mb, tempnn );
        small_descA2 = subtile_desc_create( &(descA2), m, n,
                                            dague_imin(descA2.mb, ldam), smallnb,
                                            0, 0, tempmm, tempnn );
        small_descV  = subtile_desc_create( &(descA2), m, k,
                                            dague_imin(descA2.mb, ldam), smallnb,
                                            0, 0, tempmm, descA2.nb );
        small_descT  = subtile_desc_create( &(descT2), m, k,
                                            ib, smallnb,
                                            0, 0, ib, descA2.nb );

        small_descA1->mat = A1;
        small_descA2->mat = A2;
        small_descV->mat  = V;
        small_descT->mat  = T;

        /* dague_object */
        dague_dtsmqr = dplasma_dgeqrfr_tsmqr_New( (tiled_matrix_desc_t *)small_descA1,
                                                  (tiled_matrix_desc_t *)small_descA2,
                                                  (tiled_matrix_desc_t *)small_descV,
                                                  (tiled_matrix_desc_t *)small_descT,
                                                  p_work );

        /* recursive call */
        dague_recursivecall( context, (dague_execution_context_t*)this_task,
                             dague_dtsmqr, dplasma_dgeqrfr_tsmqr_Destruct,
                             4, small_descA1, small_descA2, small_descV, small_descT );

        return DAGUE_HOOK_RETURN_ASYNC;
    }
    else
        return DAGUE_HOOK_RETURN_NEXT;
}

#line 1624 "dgeqrf_id.c"
/*-----                            END OF geqrf_dtsmqr BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
#endif  /*  defined(DAGUE_HAVE_RECURSIVE) */
static int hook_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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

/*-----                              geqrf_dtsmqr BODY                                -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 331 "dgeqrf_id.jdf"
{
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m - m * descA2.mb) : descA2.mb;
    int tempnn = (n == (descA2.nt-1)) ? (descA2.n - n * descA2.nb) : descA2.nb;
    int ldak = BLKLDD( descA1, k );
    int ldam = BLKLDD( descA2, m );
    int ldwork = ib;

#if !defined(DAGUE_DRY_RUN)
    void *p_elem_A = dague_private_memory_pop( p_work );

    CORE_dtsmqr(PlasmaLeft, PlasmaTrans,
                descA1.mb, tempnn, tempmm, tempnn, descA1.nb, ib,
                A1 /* dataA1(k,n) */, ldak,
                A2 /* dataA2(m,n) */, ldam,
                V  /* dataA2(m,k) */, ldam,
                T  /* dataT2(m,k) */, descT2.mb,
                p_elem_A, ldwork );

    dague_private_memory_push( p_work, p_elem_A );

#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 1728 "dgeqrf_id.c"
/*-----                            END OF geqrf_dtsmqr BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_geqrf_dtsmqr(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
  (void)k;  (void)mmax;  (void)m;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_geqrf_dtsmqr(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_geqrf_dtsmqr(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x7,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_geqrf_dtsmqr_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_geqrf_dtsmqr_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t assignments;
  int32_t  k, mmax, m, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &assignments);
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    assignments.mmax.value = mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &assignments);
    for( assignments.m.value = m = 0;
        assignments.m.value <= mmax;
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      for( assignments.n.value = n = (k + 1);
          assignments.n.value <= (descA1.nt - 1);
          assignments.n.value += 1, n = assignments.n.value) {
        __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
        __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
        if( !geqrf_dtsmqr_pred(assignments.k.value, assignments.mmax.value, assignments.m.value, assignments.n.value) ) continue;
        nb_tasks++;
      }
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->geqrf_dtsmqr_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->geqrf_dtsmqr_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->geqrf_dtsmqr_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_geqrf_dtsmqr_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &assignments);
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &assignments);
      assignments.mmax.value = mmax;
      __jdf2c_m_start = 0;
      __jdf2c_m_end = mmax;
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        __jdf2c_n_start = (k + 1);
        __jdf2c_n_end = (descA1.nt - 1);
        __jdf2c_n_inc = 1;
        for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
          assignments.n.value = n;
          if( !geqrf_dtsmqr_pred(k, mmax, m, n) ) continue;
          /* We did find one! Allocate the dependencies array. */
          if( dep == NULL ) {
            ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dgeqrf_id_geqrf_dtsmqr_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dgeqrf_id_geqrf_dtsmqr_m, dep, DAGUE_DEPENDENCIES_FLAG_NEXT);
          }
          if( dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min] == NULL ) {
            ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min]->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dgeqrf_id_geqrf_dtsmqr_n, dep->u.next[k-__jdf2c_k_min], DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsmqr);
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

static const __dague_chore_t __dgeqrf_id_geqrf_dtsmqr_chores[] ={
#if defined(DAGUE_HAVE_CUDA)
    { .type     = DAGUE_DEV_CUDA,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsmqr_CUDA },
#endif  /* defined(DAGUE_HAVE_CUDA) */
#if defined(DAGUE_HAVE_RECURSIVE)
    { .type     = DAGUE_DEV_RECURSIVE,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsmqr_RECURSIVE },
#endif  /* defined(DAGUE_HAVE_RECURSIVE) */
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsmqr },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_geqrf_dtsmqr = {
  .name = "geqrf_dtsmqr",
  .function_id = 5,
  .nb_flows = 4,
  .nb_parameters = 3,
  .nb_locals = 4,
  .params = { &symb_dgeqrf_id_geqrf_dtsmqr_k, &symb_dgeqrf_id_geqrf_dtsmqr_m, &symb_dgeqrf_id_geqrf_dtsmqr_n, NULL },
  .locals = { &symb_dgeqrf_id_geqrf_dtsmqr_k, &symb_dgeqrf_id_geqrf_dtsmqr_mmax, &symb_dgeqrf_id_geqrf_dtsmqr_m, &symb_dgeqrf_id_geqrf_dtsmqr_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_geqrf_dtsmqr,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = &priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr,
  .in = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T, NULL },
  .out = { &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1, &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0xf,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_geqrf_dtsmqr,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_geqrf_dtsmqr_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dgeqrf_id_geqrf_dtsmqr,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dgeqrf_id_geqrf_dtsmqr,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_geqrf_dtsmqr,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_geqrf_dtsmqr,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_geqrf_dtsmqr,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_geqrf_dtsmqr,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              geqrf_dtsmqr_out_A1                              ******/

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA1.nt - 2);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k + 1);
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA1.nt - 1);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dgeqrf_id_geqrf_dtsmqr_out_A1_inline_c_expr4_line_215(__dague_handle, locals);
}
static const expr_t expr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax = { .name = "mmax", .context_index = 2, .min = &expr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax, .max = &expr_of_symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_geqrf_dtsmqr_out_A1(__dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)n;
  (void)mmax;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, k, n);
  return 1;
}
static inline int final_data_of_dgeqrf_id_geqrf_dtsmqr_out_A1(__dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)n;
  (void)mmax;
      /** Flow of A1 */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, k, n);
    __flow_nb++;

    return __flow_nb;
}

static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep1_atline_219 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1,
};
static dague_data_t *flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep2_atline_220_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int n = assignments->n.value;
  const int mmax = assignments->mmax.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)n;
  (void)mmax;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, k, n) )
    return __ddesc->data_of(__ddesc, k, n);
  return NULL;
}

static const dep_t flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep2_atline_220 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataA1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep2_atline_220_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep1_atline_219 },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep2_atline_220 }
};

static void
iterate_predecessors_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)n;  (void)mmax;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, n);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
      const int geqrf_dtsmqr_k = k;
      if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsmqr_k;
        const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
        const int geqrf_dtsmqr_m = mmax;
        if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsmqr_m;
          const int geqrf_dtsmqr_n = n;
          if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[3].value == &ncc->locals.n.value);
            ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1_dep1_atline_219, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *eu, __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
  /* No successors, don't call iterate_successors and don't release any local deps */
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    data_repo_entry_used_once( eu, geqrf_dtsmqr_repo, this_task->data.A1.data_repo->key );
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A1.data_in) ) {  /* flow A1 */
    entry = NULL;
__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t*)&generic_locals;
  const int geqrf_dtsmqrk = target_locals->k.value = k; (void)geqrf_dtsmqrk;
  const int geqrf_dtsmqrmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, target_locals); (void)geqrf_dtsmqrmmax;
  const int geqrf_dtsmqrm = target_locals->m.value = mmax; (void)geqrf_dtsmqrm;
  const int geqrf_dtsmqrn = target_locals->n.value = n; (void)geqrf_dtsmqrn;
    entry = data_repo_lookup_entry( geqrf_dtsmqr_repo, __jdf2c_hash_geqrf_dtsmqr( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from A1:geqrf_dtsmqr to A1:geqrf_dtsmqr_out_A1");
    }
    chunk = entry->data[0];  /* A1:geqrf_dtsmqr_out_A1 <- A1:geqrf_dtsmqr */
    ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_geqrf_dtsmqr, "A1", target_locals, chunk);
      this_task->data.A1.data_in   = chunk;   /* flow A1 */
      this_task->data.A1.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A1.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** No profiling information */
  (void)k;  (void)n;  (void)mmax; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
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
  (void)k;  (void)n;  (void)mmax;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;
  (void)k;  (void)n;  (void)mmax;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA1 = this_task->data.A1.data_repo;
    if( (NULL != eA1) && (eA1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA1->sim_exec_date;
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
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            geqrf_dtsmqr_out_A1 BODY                            -----*/

#line 222 "dgeqrf_id.jdf"
{
    /* nothing */
}

#line 2289 "dgeqrf_id.c"
/*-----                        END OF geqrf_dtsmqr_out_A1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_geqrf_dtsmqr_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int n = this_task->locals.n.value;
  const int mmax = this_task->locals.mmax.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A1.data_out ) {
    this_task->data.A1.data_out->version++;  /* A1 */
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( this_task->data.A1.data_out->original != dataA1(k, n) ) {
    dague_dep_data_description_t data;
    data.data   = this_task->data.A1.data_out;
    data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
    assert( data.count > 0 );
    dague_remote_dep_memcpy(this_task->dague_handle,
                            dague_data_get_copy(dataA1(k, n), 0),
                            this_task->data.A1.data_out, &data);
  }
  (void)k;  (void)n;  (void)mmax;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_geqrf_dtsmqr_out_A1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_geqrf_dtsmqr_out_A1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_geqrf_dtsmqr_out_A1_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t assignments;
  int32_t  k, n, mmax;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_k_start, __jdf2c_k_end, __jdf2c_k_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= (descA1.nt - 2);
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    for( assignments.n.value = n = (k + 1);
        assignments.n.value <= (descA1.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      assignments.mmax.value = mmax = dgeqrf_id_geqrf_dtsmqr_out_A1_inline_c_expr4_line_215(__dague_handle, &assignments);
      if( !geqrf_dtsmqr_out_A1_pred(assignments.k.value, assignments.n.value, assignments.mmax.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->geqrf_dtsmqr_out_A1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->geqrf_dtsmqr_out_A1_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_geqrf_dtsmqr_out_A1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_k_start = 0;
    __jdf2c_k_end = (descA1.nt - 2);
    __jdf2c_k_inc = 1;
    for(k = dague_imax(__jdf2c_k_start, __jdf2c_k_min); k <= dague_imin(__jdf2c_k_end, __jdf2c_k_max); k+=__jdf2c_k_inc) {
      assignments.k.value = k;
      __jdf2c_n_start = (k + 1);
      __jdf2c_n_end = (descA1.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        mmax = dgeqrf_id_geqrf_dtsmqr_out_A1_inline_c_expr4_line_215(__dague_handle, &assignments);
        assignments.mmax.value = mmax;
        if( !geqrf_dtsmqr_out_A1_pred(k, n, mmax) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsmqr_out_A1);
  __dague_handle->super.super.dependencies_array[4] = dep;
  __dague_handle->repositories[4] = NULL;
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_n_start; (void)__jdf2c_n_end; (void)__jdf2c_n_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dgeqrf_id_geqrf_dtsmqr_out_A1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsmqr_out_A1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_geqrf_dtsmqr_out_A1 = {
  .name = "geqrf_dtsmqr_out_A1",
  .function_id = 4,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 3,
  .params = { &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k, &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n, NULL },
  .locals = { &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_k, &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_n, &symb_dgeqrf_id_geqrf_dtsmqr_out_A1_mmax, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .priority = NULL,
  .in = { &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1, NULL },
  .out = { &flow_of_dgeqrf_id_geqrf_dtsmqr_out_A1_for_A1, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_geqrf_dtsmqr_out_A1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_geqrf_dtsmqr_out_A1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)NULL,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_geqrf_dtsmqr_out_A1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                  geqrf_dtsqrt                                  ******/

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsqrt_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_dgeqrf_id_geqrf_dtsqrt_mmax_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, locals);
}
static const expr_t expr_of_symb_dgeqrf_id_geqrf_dtsqrt_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dgeqrf_id_geqrf_dtsqrt_mmax_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsqrt_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dgeqrf_id_geqrf_dtsqrt_mmax, .max = &expr_of_symb_dgeqrf_id_geqrf_dtsqrt_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int mmax = locals->mmax.value;

  (void)__dague_handle; (void)locals;
  return mmax;
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsqrt_m = { .name = "m", .context_index = 2, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_geqrf_dtsqrt(__dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
static inline int initial_data_of_dgeqrf_id_geqrf_dtsqrt(__dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
      /** Flow of A1 */
    /** Flow of A2 */
    /** Flow of T */
    __d = (dague_ddesc_t*)__dague_handle->super.dataT2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static inline int final_data_of_dgeqrf_id_geqrf_dtsqrt(__dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
      /** Flow of A1 */
    /** Flow of A2 */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;
    /** Flow of T */
    __d = (dague_ddesc_t*)__dague_handle->super.dataT2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, k);
    __flow_nb++;

    return __flow_nb;
}

static inline int priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (((descA1.mt - k) * (descA1.mt - k)) * (descA1.mt - k));
}
static const expr_t priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct }
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == 0);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue,  /* (m == 0) */
  .ctl_gather_nb = NULL,
  .function_id = 0, /* dgeqrf_id_A1_in */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_A1_in_for_A,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return !(m == 0);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse,  /* !(m == 0) */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m == mmax);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116,  /* (m == mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 2, /* dgeqrf_id_geqrf_dtsqrt_out_A1 */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int mmax = locals->mmax.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (m != mmax);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117,  /* (m != mmax) */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue,
 &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116,
 &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117 }
};

static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int k = locals->k.value;
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return ((!optid && (k == 0)) || (optid && (k == m)));
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119,  /* ((!optid && (k == 0)) || (optid && (k == m))) */
  .ctl_gather_nb = NULL,
  .function_id = 1, /* dgeqrf_id_A2_in */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_A2_in_for_A,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep2_atline_120 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
  .dep_index = 2,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k < (descA2.nt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131,  /* (k < (descA2.nt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_V,
  .dep_index = 2,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
};
static dague_data_t *flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep4_atline_132_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments)
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

static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep4_atline_132 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataA2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep4_atline_132_direct_access,
  .dep_index = 3,
  .dep_datatype_index = 2,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2 = {
  .name               = "A2",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 1,
  .flow_datatype_mask = 0x4,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119,
 &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep2_atline_120 },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131,
 &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep4_atline_132 }
};

static dague_data_t *flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep1_atline_134_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments)
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

static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep1_atline_134 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataT2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep1_atline_134_direct_access,
  .dep_index = 3,
  .dep_datatype_index = 3,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T,
};
static dague_data_t *flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep2_atline_135_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *assignments)
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

static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep2_atline_135 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataT2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep2_atline_135_direct_access,
  .dep_index = 4,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *locals)
{
  const int k = locals->k.value;

  (void)__dague_handle; (void)locals;
  return (k < (descA2.nt - 1));
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136_fct }
};
static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136,  /* (k < (descA2.nt - 1)) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_T,
  .dep_index = 5,
  .dep_datatype_index = 4,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_for_T = {
  .name               = "T",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW|FLOW_HAS_IN_DEPS,
  .flow_index         = 2,
  .flow_datatype_mask = 0x10,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep1_atline_134 },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep2_atline_135,
 &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136 }
};

static void
iterate_successors_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
  if( action_mask & 0x3 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == mmax) ) {
      __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt_out_A1.function_id];
        const int geqrf_dtsqrt_out_A1_k = k;
        if( (geqrf_dtsqrt_out_A1_k >= (0)) && (geqrf_dtsqrt_out_A1_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_out_A1_k;
          const int geqrf_dtsqrt_out_A1_mmax = dgeqrf_id_geqrf_dtsqrt_out_A1_inline_c_expr6_line_92(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_out_A1_mmax;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.k.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority;
        RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep2_atline_116, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (m != mmax) ) {
      __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
        const int geqrf_dtsqrt_k = k;
        if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_k;
          const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
          const int geqrf_dtsqrt_m = (m + 1);
          if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep3_atline_117, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  }
  if( action_mask & 0xc ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x4 ) {
        if( (k < (descA2.nt - 1)) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = k;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = m;
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            int geqrf_dtsmqr_n;
          for( geqrf_dtsmqr_n = (k + 1);geqrf_dtsmqr_n <= (descA2.nt - 1); geqrf_dtsmqr_n+=1) {
              if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.n.value);
                ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
                if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
                nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
              RELEASE_DEP_OUTPUT(eu, "A2", this_task, "V", &nc, rank_src, rank_dst, &data);
              if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep3_atline_131, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
    /* action_mask & 0x8 goes to data dataA2(m, k) */
  }
  if( action_mask & 0x30 ) {  /* Flow of Data T */
    data.data   = this_task->data.T.data_out;
    /* action_mask & 0x10 goes to data dataT2(m, k) */
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x20 ) {
        if( (k < (descA2.nt - 1)) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = k;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = m;
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            int geqrf_dtsmqr_n;
          for( geqrf_dtsmqr_n = (k + 1);geqrf_dtsmqr_n <= (descA2.nt - 1); geqrf_dtsmqr_n+=1) {
              if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
                assert(&nc.locals[3].value == &ncc->locals.n.value);
                ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
                rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
                if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                  vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
                nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
              RELEASE_DEP_OUTPUT(eu, "T", this_task, "T", &nc, rank_src, rank_dst, &data);
              if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T_dep3_atline_136, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
                }
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static void
iterate_predecessors_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
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
  if( action_mask & 0x1 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == 0) ) {
      __dague_dgeqrf_id_A1_in_task_t* ncc = (__dague_dgeqrf_id_A1_in_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_A1_in.function_id];
        const int A1_in_m = k;
        if( (A1_in_m >= (0)) && (A1_in_m <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = A1_in_m;
          const int A1_in_n = k;
          if( (A1_in_n >= (ncc->locals.m.value)) && (A1_in_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[1].value == &ncc->locals.n.value);
            ncc->locals.n.value = A1_in_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iftrue, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    } else {
      __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
        const int geqrf_dtsqrt_k = k;
        if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_k;
          const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
          const int geqrf_dtsqrt_m = (m - 1);
          if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1_dep1_atline_115_iffalse, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  }
  if( action_mask & 0x6 ) {  /* Flow of Data A2 */
    data.data   = this_task->data.A2.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x2 ) {
        if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
      __dague_dgeqrf_id_A2_in_task_t* ncc = (__dague_dgeqrf_id_A2_in_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_A2_in.function_id];
        const int A2_in_m = m;
        if( (A2_in_m >= (0)) && (A2_in_m <= ((descA2.mt - 1))) ) {
          assert(&nc.locals[0].value == &ncc->locals.m.value);
          ncc->locals.m.value = A2_in_m;
          const int A2_in_nmin = (optid ? ncc->locals.m.value : 0);
          assert(&nc.locals[1].value == &ncc->locals.nmin.value);
          ncc->locals.nmin.value = A2_in_nmin;
          const int A2_in_n = k;
          if( (A2_in_n >= (ncc->locals.nmin.value)) && (A2_in_n <= ((descA2.nt - 1))) ) {
            assert(&nc.locals[2].value == &ncc->locals.n.value);
            ncc->locals.n.value = A2_in_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority;
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep1_atline_119, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x4 ) {
        __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
      const int geqrf_dtsmqr_k = (k - 1);
      if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsmqr_k;
        const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
        const int geqrf_dtsmqr_m = m;
        if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsmqr_m;
          const int geqrf_dtsmqr_n = k;
          if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
            assert(&nc.locals[3].value == &ncc->locals.n.value);
            ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A2", this_task, "A2", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2_dep2_atline_120, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
        }
  }
  }
  /* Flow of data T has only OUTPUT dependencies to Memory */
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *eu, __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
    arg.output_entry = data_repo_lookup_entry_and_create( eu, geqrf_dtsqrt_repo, __jdf2c_hash_geqrf_dtsqrt(__dague_handle, (__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dgeqrf_id_geqrf_dtsqrt(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(geqrf_dtsqrt_repo, arg.output_entry->key, arg.output_usage);
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

    (void)k; (void)mmax; (void)m;

    if( (m == 0) ) {
      data_repo_entry_used_once( eu, A1_in_repo, this_task->data.A1.data_repo->key );
    } else {
      data_repo_entry_used_once( eu, geqrf_dtsqrt_repo, this_task->data.A1.data_repo->key );
    }
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
    if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
      data_repo_entry_used_once( eu, A2_in_repo, this_task->data.A2.data_repo->key );
    }
    else {
    data_repo_entry_used_once( eu, geqrf_dtsmqr_repo, this_task->data.A2.data_repo->key );
    }
    if( NULL != this_task->data.A2.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A2.data_in);
    }
    if( NULL != this_task->data.T.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.T.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A1.data_in) ) {  /* flow A1 */
    entry = NULL;
    if( (m == 0) ) {
__dague_dgeqrf_id_A1_in_assignment_t *target_locals = (__dague_dgeqrf_id_A1_in_assignment_t*)&generic_locals;
      const int A1_inm = target_locals->m.value = k; (void)A1_inm;
      const int A1_inn = target_locals->n.value = k; (void)A1_inn;
      entry = data_repo_lookup_entry( A1_in_repo, __jdf2c_hash_A1_in( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:A1_in to A1:geqrf_dtsqrt");
      }
      chunk = entry->data[0];  /* A1:geqrf_dtsqrt <- A:A1_in */
      ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_A1_in, "A", target_locals, chunk);
    } else {
__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t*)&generic_locals;
      const int geqrf_dtsqrtk = target_locals->k.value = k; (void)geqrf_dtsqrtk;
      const int geqrf_dtsqrtmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, target_locals); (void)geqrf_dtsqrtmmax;
      const int geqrf_dtsqrtm = target_locals->m.value = (m - 1); (void)geqrf_dtsqrtm;
      entry = data_repo_lookup_entry( geqrf_dtsqrt_repo, __jdf2c_hash_geqrf_dtsqrt( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A1:geqrf_dtsqrt to A1:geqrf_dtsqrt");
      }
      chunk = entry->data[0];  /* A1:geqrf_dtsqrt <- A1:geqrf_dtsqrt */
      ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_geqrf_dtsqrt, "A1", target_locals, chunk);
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
    if( ((!optid && (k == 0)) || (optid && (k == m))) ) {
__dague_dgeqrf_id_A2_in_assignment_t *target_locals = (__dague_dgeqrf_id_A2_in_assignment_t*)&generic_locals;
      const int A2_inm = target_locals->m.value = m; (void)A2_inm;
      const int A2_innmin = target_locals->nmin.value = (optid ? A2_inm : 0); (void)A2_innmin;
      const int A2_inn = target_locals->n.value = k; (void)A2_inn;
      entry = data_repo_lookup_entry( A2_in_repo, __jdf2c_hash_A2_in( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A:A2_in to A2:geqrf_dtsqrt");
      }
      chunk = entry->data[0];  /* A2:geqrf_dtsqrt <- A:A2_in */
      ACQUIRE_FLOW(this_task, "A2", &dgeqrf_id_A2_in, "A", target_locals, chunk);
    }
    else {
__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsmqr_assignment_t*)&generic_locals;
      const int geqrf_dtsmqrk = target_locals->k.value = (k - 1); (void)geqrf_dtsmqrk;
      const int geqrf_dtsmqrmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, target_locals); (void)geqrf_dtsmqrmmax;
      const int geqrf_dtsmqrm = target_locals->m.value = m; (void)geqrf_dtsmqrm;
      const int geqrf_dtsmqrn = target_locals->n.value = k; (void)geqrf_dtsmqrn;
      entry = data_repo_lookup_entry( geqrf_dtsmqr_repo, __jdf2c_hash_geqrf_dtsmqr( __dague_handle, target_locals ));
      if ( NULL == entry ) {
          dague_abort("A NULL has been forwarded from A2:geqrf_dtsmqr to A2:geqrf_dtsqrt");
      }
      chunk = entry->data[1];  /* A2:geqrf_dtsqrt <- A2:geqrf_dtsmqr */
      ACQUIRE_FLOW(this_task, "A2", &dgeqrf_id_geqrf_dtsmqr, "A2", target_locals, chunk);
    }
      this_task->data.A2.data_in   = chunk;   /* flow A2 */
      this_task->data.A2.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A2.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  if( NULL == (chunk = this_task->data.T.data_in) ) {  /* flow T */
    entry = NULL;
    chunk = dague_data_get_copy(dataT2(m, k), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.T.data_in   = chunk;   /* flow T */
      this_task->data.T.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.T.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataA2;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataA2))->data_key((dague_ddesc_t*)__dague_handle->super.dataA2, m, k);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)k;  (void)mmax;  (void)m; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
if( (*flow_mask) & 0x2U ) {  /* Flow A2 */
    if( ((*flow_mask) & 0x2U) && (((!optid && (k == 0)) || (optid && (k == m)))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x2U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x2U) */
if( (*flow_mask) & 0x4U ) {  /* Flow T */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x4U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x4U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow A1 */
    if( ((*flow_mask) & 0x3U) && ((m == mmax) || (m != mmax)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x3U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x3U) */
if( (*flow_mask) & 0xcU ) {  /* Flow A2 */
    if( ((*flow_mask) & 0xcU) && ((k < (descA2.nt - 1))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0xcU;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0xcU) */
if( (*flow_mask) & 0x30U ) {  /* Flow T */
    if( ((*flow_mask) & 0x30U) && ((k < (descA2.nt - 1))) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x30U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
}  /* (flow_mask & 0x30U) */
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
#if defined(DAGUE_HAVE_RECURSIVE)
static int hook_of_dgeqrf_id_geqrf_dtsqrt_RECURSIVE(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)mmax;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;
  dague_data_copy_t *gA2 = this_task->data.A2.data_in;
  void *A2 = (NULL != gA2) ? DAGUE_DATA_COPY_GET_PTR(gA2) : NULL; (void)A2;
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
    if ( NULL != gT ) {
      dague_data_transfer_ownership_to_copy( gT->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
  cache_buf_referenced(context->closest_cache, A2);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              geqrf_dtsqrt BODY                                -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 142 "dgeqrf_id.jdf"
{
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m - m * descA2.mb) : descA2.mb;
    int tempkn = (k == (descA2.nt-1)) ? (descA2.n - k * descA2.nb) : descA2.nb;
    int ldak = BLKLDD( descA2, k );
    int ldam = BLKLDD( descA2, m );

    if (tempmm > smallnb) {
        subtile_desc_t *small_descA1;
        subtile_desc_t *small_descA2;
        subtile_desc_t *small_descT;
        dague_handle_t *dague_dtsqrt;


        small_descA1 = subtile_desc_create( &(descA2), k, k,
                                            dague_imin(descA2.mb, ldak), smallnb,
                                            0, 0, tempkn, tempkn );
        small_descA2 = subtile_desc_create( &(descA2), m, k,
                                            dague_imin(descA2.mb, ldam), smallnb,
                                            0, 0, tempmm, tempkn );
        small_descT  = subtile_desc_create( &(descT2), m, k,
                                            ib, smallnb,
                                            0, 0, ib, tempkn );

        small_descA1->mat = A1;
        small_descA2->mat = A2;
        small_descT->mat  = T;

        /* dague_object */
        dague_dtsqrt = dplasma_dgeqrfr_tsqrt_New((tiled_matrix_desc_t *)small_descA1,
                                                 (tiled_matrix_desc_t *)small_descA2,
                                                 (tiled_matrix_desc_t *)small_descT,
                                                 p_work, p_tau );

        /* recursive call */
        dague_recursivecall( context, (dague_execution_context_t*)this_task,
                             dague_dtsqrt, dplasma_dgeqrfr_tsqrt_Destruct,
                             3, small_descA1, small_descA2, small_descT );

        return DAGUE_HOOK_RETURN_ASYNC;
    }
    else
        return DAGUE_HOOK_RETURN_NEXT;
}

#line 3560 "dgeqrf_id.c"
/*-----                            END OF geqrf_dtsqrt BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
#endif  /*  defined(DAGUE_HAVE_RECURSIVE) */
static int hook_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
  (void)k;  (void)mmax;  (void)m;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;
  dague_data_copy_t *gA2 = this_task->data.A2.data_in;
  void *A2 = (NULL != gA2) ? DAGUE_DATA_COPY_GET_PTR(gA2) : NULL; (void)A2;
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
    if ( NULL != gT ) {
      dague_data_transfer_ownership_to_copy( gT->original, 0 /* device */,
                                           FLOW_ACCESS_RW);
    }
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
  cache_buf_referenced(context->closest_cache, A2);
  cache_buf_referenced(context->closest_cache, T);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                              geqrf_dtsqrt BODY                                -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 188 "dgeqrf_id.jdf"
{
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m - m * descA2.mb) : descA2.mb;
    int tempkn = (k == (descA2.nt-1)) ? (descA2.n - k * descA2.nb) : descA2.nb;
    int ldak = BLKLDD( descA2, k );
    int ldam = BLKLDD( descA2, m );

#if !defined(DAGUE_DRY_RUN)

    void *p_elem_A = dague_private_memory_pop( p_tau );
    void *p_elem_B = dague_private_memory_pop( p_work );

    CORE_dtsqrt(tempmm, tempkn, ib,
                A1 /* dataA1(k,k) */, ldak,
                A2 /* dataA2(m,k) */, ldam,
                T  /* dataT2(m,k) */, descT2.mb,
                p_elem_A, p_elem_B );

    dague_private_memory_push( p_tau,  p_elem_A );
    dague_private_memory_push( p_work, p_elem_B );

#endif  /* !defined(DAGUE_DRY_RUN) */
}

#line 3661 "dgeqrf_id.c"
/*-----                            END OF geqrf_dtsqrt BODY                            -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_geqrf_dtsqrt(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  const int m = this_task->locals.m.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A1.data_out ) {
    this_task->data.A1.data_out->version++;  /* A1 */
  }
  if ( NULL != this_task->data.A2.data_out ) {
    this_task->data.A2.data_out->version++;  /* A2 */
  }
  if ( NULL != this_task->data.T.data_out ) {
    this_task->data.T.data_out->version++;  /* T */
  }
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( this_task->data.A2.data_out->original != dataA2(m, k) ) {
    dague_dep_data_description_t data;
    data.data   = this_task->data.A2.data_out;
    data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
    assert( data.count > 0 );
    dague_remote_dep_memcpy(this_task->dague_handle,
                            dague_data_get_copy(dataA2(m, k), 0),
                            this_task->data.A2.data_out, &data);
  }
  if( this_task->data.T.data_out->original != dataT2(m, k) ) {
    dague_dep_data_description_t data;
    data.data   = this_task->data.T.data_out;
    data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_LITTLE_T_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
    assert( data.count > 0 );
    dague_remote_dep_memcpy(this_task->dague_handle,
                            dague_data_get_copy(dataT2(m, k), 0),
                            this_task->data.T.data_out, &data);
  }
  (void)k;  (void)mmax;  (void)m;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_geqrf_dtsqrt(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_geqrf_dtsqrt(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x3f,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_geqrf_dtsqrt_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_geqrf_dtsqrt_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t assignments;
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
    assignments.mmax.value = mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &assignments);
    for( assignments.m.value = m = 0;
        assignments.m.value <= mmax;
        assignments.m.value += 1, m = assignments.m.value) {
      __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
      __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
      if( !geqrf_dtsqrt_pred(assignments.k.value, assignments.mmax.value, assignments.m.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->geqrf_dtsqrt_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  __dague_handle->geqrf_dtsqrt_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_geqrf_dtsqrt_internal_init (nb_tasks = %d)", nb_tasks);
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
      mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &assignments);
      assignments.mmax.value = mmax;
      __jdf2c_m_start = 0;
      __jdf2c_m_end = mmax;
      __jdf2c_m_inc = 1;
      for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
        assignments.m.value = m;
        if( !geqrf_dtsqrt_pred(k, mmax, m) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dgeqrf_id_geqrf_dtsqrt_k, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[k-__jdf2c_k_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[k-__jdf2c_k_min], __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dgeqrf_id_geqrf_dtsqrt_m, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsqrt);
  __dague_handle->super.super.dependencies_array[3] = dep;
  __dague_handle->repositories[3] = data_repo_create_nothreadsafe(nb_tasks, 3);
  (void)__jdf2c_k_start; (void)__jdf2c_k_end; (void)__jdf2c_k_inc;  (void)__jdf2c_m_start; (void)__jdf2c_m_end; (void)__jdf2c_m_inc;  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dgeqrf_id_geqrf_dtsqrt_chores[] ={
#if defined(DAGUE_HAVE_RECURSIVE)
    { .type     = DAGUE_DEV_RECURSIVE,
      .dyld     = NULL,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsqrt_RECURSIVE },
#endif  /* defined(DAGUE_HAVE_RECURSIVE) */
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsqrt },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_geqrf_dtsqrt = {
  .name = "geqrf_dtsqrt",
  .function_id = 3,
  .nb_flows = 3,
  .nb_parameters = 2,
  .nb_locals = 3,
  .params = { &symb_dgeqrf_id_geqrf_dtsqrt_k, &symb_dgeqrf_id_geqrf_dtsqrt_m, NULL },
  .locals = { &symb_dgeqrf_id_geqrf_dtsqrt_k, &symb_dgeqrf_id_geqrf_dtsqrt_mmax, &symb_dgeqrf_id_geqrf_dtsqrt_m, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_geqrf_dtsqrt,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dgeqrf_id_geqrf_dtsqrt,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dgeqrf_id_geqrf_dtsqrt,
  .priority = &priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr,
  .in = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T, NULL },
  .out = { &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2, &flow_of_dgeqrf_id_geqrf_dtsqrt_for_T, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x7,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_geqrf_dtsqrt,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_geqrf_dtsqrt_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dgeqrf_id_geqrf_dtsqrt,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dgeqrf_id_geqrf_dtsqrt,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_geqrf_dtsqrt,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_geqrf_dtsqrt,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_geqrf_dtsqrt,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_geqrf_dtsqrt,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                              geqrf_dtsqrt_out_A1                              ******/

static inline int minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k = { .name = "k", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k, .max = &maxexpr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int expr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return dgeqrf_id_geqrf_dtsqrt_out_A1_inline_c_expr6_line_92(__dague_handle, locals);
}
static const expr_t expr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax_fct }
};
static const symbol_t symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax = { .name = "mmax", .context_index = 1, .min = &expr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax, .max = &expr_of_symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_geqrf_dtsqrt_out_A1(__dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)k;
  (void)mmax;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, k, k);
  return 1;
}
static inline int final_data_of_dgeqrf_id_geqrf_dtsqrt_out_A1(__dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)k;
  (void)mmax;
      /** Flow of A1 */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, k, k);
    __flow_nb++;

    return __flow_nb;
}

static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep1_atline_96 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1,
};
static dague_data_t *flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep2_atline_97_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int k = assignments->k.value;
  const int mmax = assignments->mmax.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)k;
  (void)mmax;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, k, k) )
    return __ddesc->data_of(__ddesc, k, k);
  return NULL;
}

static const dep_t flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep2_atline_97 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataA1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep2_atline_97_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1,
};
static const dague_flow_t flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1 = {
  .name               = "A1",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep1_atline_96 },
  .dep_out    = { &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep2_atline_97 }
};

static void
iterate_predecessors_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)k;  (void)mmax;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, k, k);
#endif
  if( action_mask & 0x1 ) {  /* Flow of Data A1 */
    data.data   = this_task->data.A1.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
    nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
      const int geqrf_dtsqrt_k = k;
      if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
        assert(&nc.locals[0].value == &ncc->locals.k.value);
        ncc->locals.k.value = geqrf_dtsqrt_k;
        const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
        assert(&nc.locals[1].value == &ncc->locals.mmax.value);
        ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
        const int geqrf_dtsqrt_m = mmax;
        if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
          assert(&nc.locals[2].value == &ncc->locals.m.value);
          ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
          rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
            vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
          nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
        RELEASE_DEP_OUTPUT(eu, "A1", this_task, "A1", &nc, rank_src, rank_dst, &data);
        if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1_dep1_atline_96, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
          }
        }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *eu, __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
  /* No successors, don't call iterate_successors and don't release any local deps */
  if( action_mask & DAGUE_ACTION_RELEASE_LOCAL_REFS ) {
    data_repo_entry_used_once( eu, geqrf_dtsqrt_repo, this_task->data.A1.data_repo->key );
    if( NULL != this_task->data.A1.data_in ) {
        DAGUE_DATA_COPY_RELEASE(this_task->data.A1.data_in);
    }
  }
  return 0;
}

static int data_lookup_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A1.data_in) ) {  /* flow A1 */
    entry = NULL;
__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t *target_locals = (__dague_dgeqrf_id_geqrf_dtsqrt_assignment_t*)&generic_locals;
  const int geqrf_dtsqrtk = target_locals->k.value = k; (void)geqrf_dtsqrtk;
  const int geqrf_dtsqrtmmax = target_locals->mmax.value = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, target_locals); (void)geqrf_dtsqrtmmax;
  const int geqrf_dtsqrtm = target_locals->m.value = mmax; (void)geqrf_dtsqrtm;
    entry = data_repo_lookup_entry( geqrf_dtsqrt_repo, __jdf2c_hash_geqrf_dtsqrt( __dague_handle, target_locals ));
    if ( NULL == entry ) {
        dague_abort("A NULL has been forwarded from A1:geqrf_dtsqrt to A1:geqrf_dtsqrt_out_A1");
    }
    chunk = entry->data[0];  /* A1:geqrf_dtsqrt_out_A1 <- A1:geqrf_dtsqrt */
    ACQUIRE_FLOW(this_task, "A1", &dgeqrf_id_geqrf_dtsqrt, "A1", target_locals, chunk);
      this_task->data.A1.data_in   = chunk;   /* flow A1 */
      this_task->data.A1.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A1.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** No profiling information */
  (void)k;  (void)mmax; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *eu, const __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A1 */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
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
  (void)k;  (void)mmax;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
  (void)k;  (void)mmax;

  /** Declare the variables that will hold the data, and all the accounting for each */
  dague_data_copy_t *gA1 = this_task->data.A1.data_in;
  void *A1 = (NULL != gA1) ? DAGUE_DATA_COPY_GET_PTR(gA1) : NULL; (void)A1;

  /** Update starting simulation date */
#if defined(DAGUE_SIM)
  {
    this_task->sim_exec_date = 0;
    data_repo_entry_t *eA1 = this_task->data.A1.data_repo;
    if( (NULL != eA1) && (eA1->sim_exec_date > this_task->sim_exec_date) )
      this_task->sim_exec_date = eA1->sim_exec_date;
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
#endif  /* defined(DAGUE_HAVE_CUDA) */
  /** Cache Awareness Accounting */
#if defined(DAGUE_CACHE_AWARENESS)
  cache_buf_referenced(context->closest_cache, A1);
#endif /* DAGUE_CACHE_AWARENESS */


#if !defined(DAGUE_PROF_DRY_BODY)

/*-----                            geqrf_dtsqrt_out_A1 BODY                            -----*/

#line 99 "dgeqrf_id.jdf"
{
    /* nothing */
}

#line 4185 "dgeqrf_id.c"
/*-----                        END OF geqrf_dtsqrt_out_A1 BODY                        -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_geqrf_dtsqrt_out_A1(dague_execution_unit_t *context, __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int k = this_task->locals.k.value;
  const int mmax = this_task->locals.mmax.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A1.data_out ) {
    this_task->data.A1.data_out->version++;  /* A1 */
  }
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  if( this_task->data.A1.data_out->original != dataA1(k, k) ) {
    dague_dep_data_description_t data;
    data.data   = this_task->data.A1.data_out;
    data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
    assert( data.count > 0 );
    dague_remote_dep_memcpy(this_task->dague_handle,
                            dague_data_get_copy(dataA1(k, k), 0),
                            this_task->data.A1.data_out, &data);
  }
  (void)k;  (void)mmax;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_geqrf_dtsqrt_out_A1(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_geqrf_dtsqrt_out_A1(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x1,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_geqrf_dtsqrt_out_A1_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t assignments;
  int32_t  k, mmax;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_k_min = 0x7fffffff;
  int32_t __jdf2c_k_max = 0;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.k.value = k = 0;
      assignments.k.value <= KT;
      assignments.k.value += 1, k = assignments.k.value) {
    __jdf2c_k_max = dague_imax(__jdf2c_k_max, assignments.k.value);
    __jdf2c_k_min = dague_imin(__jdf2c_k_min, assignments.k.value);
    assignments.mmax.value = mmax = dgeqrf_id_geqrf_dtsqrt_out_A1_inline_c_expr6_line_92(__dague_handle, &assignments);
    if( !geqrf_dtsqrt_out_A1_pred(assignments.k.value, assignments.mmax.value) ) continue;
    nb_tasks++;
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->geqrf_dtsqrt_out_A1_k_range = (__jdf2c_k_max - __jdf2c_k_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_geqrf_dtsqrt_out_A1_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    ALLOCATE_DEP_TRACKING(dep, __jdf2c_k_min, __jdf2c_k_max, "k", &symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k, NULL, DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsqrt_out_A1);
  __dague_handle->super.super.dependencies_array[2] = dep;
  __dague_handle->repositories[2] = NULL;
  (void)assignments; (void)__dague_handle; (void)eu;
  if(0 == dague_atomic_dec_32b(&__dague_handle->sync_point)) {
    dague_handle_enable((dague_handle_t*)__dague_handle, &__dague_handle->startup_queue,
                        (dague_execution_context_t*)this_task, eu, __dague_handle->super.super.nb_pending_actions);
    return DAGUE_HOOK_RETURN_DONE;
  }
  return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dgeqrf_id_geqrf_dtsqrt_out_A1_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_geqrf_dtsqrt_out_A1 },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_geqrf_dtsqrt_out_A1 = {
  .name = "geqrf_dtsqrt_out_A1",
  .function_id = 2,
  .nb_flows = 1,
  .nb_parameters = 1,
  .nb_locals = 2,
  .params = { &symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k, NULL },
  .locals = { &symb_dgeqrf_id_geqrf_dtsqrt_out_A1_k, &symb_dgeqrf_id_geqrf_dtsqrt_out_A1_mmax, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .initial_data = (dague_data_ref_fn_t*)NULL,
  .final_data = (dague_data_ref_fn_t*)final_data_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .priority = NULL,
  .in = { &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1, NULL },
  .out = { &flow_of_dgeqrf_id_geqrf_dtsqrt_out_A1_for_A1, NULL },
  .flags = 0x0 | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_geqrf_dtsqrt_out_A1,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_geqrf_dtsqrt_out_A1_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)NULL,
  .iterate_predecessors = (dague_traverse_function_t*)iterate_predecessors_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_geqrf_dtsqrt_out_A1,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                    A2_in                                      ******/

static inline int minexpr_of_symb_dgeqrf_id_A2_in_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_A2_in_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_A2_in_m_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_A2_in_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA2.mt - 1);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_A2_in_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_A2_in_m_fct }
};
static const symbol_t symb_dgeqrf_id_A2_in_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_A2_in_m, .max = &maxexpr_of_symb_dgeqrf_id_A2_in_m, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int expr_of_symb_dgeqrf_id_A2_in_nmin_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return (optid ? m : 0);
}
static const expr_t expr_of_symb_dgeqrf_id_A2_in_nmin = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_symb_dgeqrf_id_A2_in_nmin_fct }
};
static const symbol_t symb_dgeqrf_id_A2_in_nmin = { .name = "nmin", .context_index = 1, .min = &expr_of_symb_dgeqrf_id_A2_in_nmin, .max = &expr_of_symb_dgeqrf_id_A2_in_nmin, .cst_inc = 0, .expr_inc = NULL,  .flags = 0x0};

static inline int minexpr_of_symb_dgeqrf_id_A2_in_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{
  const int nmin = locals->nmin.value;

  (void)__dague_handle; (void)locals;
  return nmin;
}
static const expr_t minexpr_of_symb_dgeqrf_id_A2_in_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_A2_in_n_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_A2_in_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA2.nt - 1);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_A2_in_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_A2_in_n_fct }
};
static const symbol_t symb_dgeqrf_id_A2_in_n = { .name = "n", .context_index = 2, .min = &minexpr_of_symb_dgeqrf_id_A2_in_n, .max = &maxexpr_of_symb_dgeqrf_id_A2_in_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_A2_in(__dague_dgeqrf_id_A2_in_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)m;
  (void)nmin;
  (void)n;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA2;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int initial_data_of_dgeqrf_id_A2_in(__dague_dgeqrf_id_A2_in_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)m;
  (void)nmin;
  (void)n;
      /** Flow of A */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA2;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, n);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dgeqrf_id_A2_in_for_A_dep1_atline_68_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int m = assignments->m.value;
  const int nmin = assignments->nmin.value;
  const int n = assignments->n.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)m;
  (void)nmin;
  (void)n;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA2;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dgeqrf_id_A2_in_for_A_dep1_atline_68 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataA2 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_A2_in_for_A_dep1_atline_68_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_A2_in_for_A,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{
  const int nmin = locals->nmin.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (n == nmin);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69_fct }
};
static const dep_t flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69,  /* (n == nmin) */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A2,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_A2_in_for_A,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A2_in_assignment_t *locals)
{
  const int nmin = locals->nmin.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (n != nmin);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70_fct }
};
static const dep_t flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70,  /* (n != nmin) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A2,
  .dep_index = 1,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_A2_in_for_A,
};
static const dague_flow_t flow_of_dgeqrf_id_A2_in_for_A = {
  .name               = "A",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_RW|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x1,
  .dep_in     = { &flow_of_dgeqrf_id_A2_in_for_A_dep1_atline_68 },
  .dep_out    = { &flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69,
 &flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70 }
};

static void
iterate_successors_of_dgeqrf_id_A2_in(dague_execution_unit_t *eu, const __dague_dgeqrf_id_A2_in_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)m;  (void)nmin;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (n == nmin) ) {
      __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
        const int geqrf_dtsqrt_k = nmin;
        if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_k;
          const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
          const int geqrf_dtsqrt_m = m;
          if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A", this_task, "A2", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_A2_in_for_A_dep2_atline_69, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
  if( action_mask & 0x2 ) {
        if( (n != nmin) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = nmin;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = m;
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            const int geqrf_dtsmqr_n = n;
            if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
            RELEASE_DEP_OUTPUT(eu, "A", this_task, "A2", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_A2_in_for_A_dep3_atline_70, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_A2_in(dague_execution_unit_t *eu, __dague_dgeqrf_id_A2_in_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
    arg.output_entry = data_repo_lookup_entry_and_create( eu, A2_in_repo, __jdf2c_hash_A2_in(__dague_handle, (__dague_dgeqrf_id_A2_in_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dgeqrf_id_A2_in(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(A2_in_repo, arg.output_entry->key, arg.output_usage);
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

static int data_lookup_of_dgeqrf_id_A2_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A2_in_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
    chunk = dague_data_get_copy(dataA2(m, n), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL != chunk ) {
        this_task->data.A.data_out = dague_data_get_copy(chunk->original, target_device);
    }
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataA2;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataA2))->data_key((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)m;  (void)nmin;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_A2_in(dague_execution_unit_t *eu, const __dague_dgeqrf_id_A2_in_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow A */
    if( ((*flow_mask) & 0x3U) && ((n == nmin) || (n != nmin)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
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
  (void)m;  (void)nmin;  (void)n;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dgeqrf_id_A2_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A2_in_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;
  (void)m;  (void)nmin;  (void)n;

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

/*-----                                  A2_in BODY                                  -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 73 "dgeqrf_id.jdf"
{
    int tempmm = (m == (descA2.mt-1)) ? (descA2.m - m * descA2.mb) : descA2.mb;
    int tempnn = (n == (descA2.nt-1)) ? (descA2.n - n * descA2.nb) : descA2.nb;
    int ldam = BLKLDD( descA2, m );

    if (m == n) {
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, 'A', tempmm, tempnn,
            0., 1., A, ldam );
    } else {
        LAPACKE_dlaset_work(
            LAPACK_COL_MAJOR, 'A', tempmm, tempnn,
            0., 0., A, ldam );
    }
}

#line 4796 "dgeqrf_id.c"
/*-----                              END OF A2_in BODY                                -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_A2_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A2_in_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int m = this_task->locals.m.value;
  const int nmin = this_task->locals.nmin.value;
  const int n = this_task->locals.n.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  if ( NULL != this_task->data.A.data_out ) {
    this_task->data.A.data_out->version++;  /* A */
  }
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)m;  (void)nmin;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_A2_in(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_A2_in(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x3,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_A2_in_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_A2_in_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_A2_in_assignment_t assignments;
  int32_t  m, nmin, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.m.value = m = 0;
      assignments.m.value <= (descA2.mt - 1);
      assignments.m.value += 1, m = assignments.m.value) {
    __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
    __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
    assignments.nmin.value = nmin = (optid ? m : 0);
    for( assignments.n.value = n = nmin;
        assignments.n.value <= (descA2.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      if( !A2_in_pred(assignments.m.value, assignments.nmin.value, assignments.n.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->A2_in_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->A2_in_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_A2_in_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_m_start = 0;
    __jdf2c_m_end = (descA2.mt - 1);
    __jdf2c_m_inc = 1;
    for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
      assignments.m.value = m;
      nmin = (optid ? m : 0);
      assignments.nmin.value = nmin;
      __jdf2c_n_start = nmin;
      __jdf2c_n_end = (descA2.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        if( !A2_in_pred(m, nmin, n) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dgeqrf_id_A2_in_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-__jdf2c_m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dgeqrf_id_A2_in_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_A2_in);
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

static int __jdf2c_startup_A2_in(dague_execution_unit_t * eu, __dague_dgeqrf_id_A2_in_task_t *this_task)
{
  __dague_dgeqrf_id_A2_in_task_t* new_task;
  __dague_dgeqrf_id_internal_handle_t* __dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  int nmin = this_task->locals.nmin.value;  /* retrieve value saved during the last iteration */
  int n = this_task->locals.n.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.m.value = m = 0;
      this_task->locals.m.value <= (descA2.mt - 1);
      this_task->locals.m.value += 1, m = this_task->locals.m.value) {
    this_task->locals.nmin.value = nmin = (optid ? m : 0);
    for(this_task->locals.n.value = n = nmin;
        this_task->locals.n.value <= (descA2.nt - 1);
        this_task->locals.n.value += 1, n = this_task->locals.n.value) {
      if( !A2_in_pred(m, nmin, n) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, m, n);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dgeqrf_id_A2_in_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dgeqrf_id_A2_in.function_id];
      new_task->chore_id     = 0;
      new_task->locals.m.value = this_task->locals.m.value;
      new_task->locals.nmin.value = this_task->locals.nmin.value;
      new_task->locals.n.value = this_task->locals.n.value;
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

static const __dague_chore_t __dgeqrf_id_A2_in_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_A2_in },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_A2_in = {
  .name = "A2_in",
  .function_id = 1,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 3,
  .params = { &symb_dgeqrf_id_A2_in_m, &symb_dgeqrf_id_A2_in_n, NULL },
  .locals = { &symb_dgeqrf_id_A2_in_m, &symb_dgeqrf_id_A2_in_nmin, &symb_dgeqrf_id_A2_in_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_A2_in,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dgeqrf_id_A2_in,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dgeqrf_id_A2_in_for_A, NULL },
  .out = { &flow_of_dgeqrf_id_A2_in_for_A, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_A2_in,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_A2_in_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dgeqrf_id_A2_in,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_A2_in,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_A2_in,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_A2_in,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_A2_in,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


/******                                    A1_in                                      ******/

static inline int minexpr_of_symb_dgeqrf_id_A1_in_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{
  (void)__dague_handle; (void)locals;
  return 0;
}
static const expr_t minexpr_of_symb_dgeqrf_id_A1_in_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_A1_in_m_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_A1_in_m_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return KT;
}
static const expr_t maxexpr_of_symb_dgeqrf_id_A1_in_m = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_A1_in_m_fct }
};
static const symbol_t symb_dgeqrf_id_A1_in_m = { .name = "m", .context_index = 0, .min = &minexpr_of_symb_dgeqrf_id_A1_in_m, .max = &maxexpr_of_symb_dgeqrf_id_A1_in_m, .cst_inc = 1, .expr_inc = NULL,  .flags = DAGUE_SYMBOL_IS_STANDALONE};

static inline int minexpr_of_symb_dgeqrf_id_A1_in_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{
  const int m = locals->m.value;

  (void)__dague_handle; (void)locals;
  return m;
}
static const expr_t minexpr_of_symb_dgeqrf_id_A1_in_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)minexpr_of_symb_dgeqrf_id_A1_in_n_fct }
};
static inline int maxexpr_of_symb_dgeqrf_id_A1_in_n_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{


  (void)__dague_handle; (void)locals;
  return (descA1.nt - 1);
}
static const expr_t maxexpr_of_symb_dgeqrf_id_A1_in_n = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)maxexpr_of_symb_dgeqrf_id_A1_in_n_fct }
};
static const symbol_t symb_dgeqrf_id_A1_in_n = { .name = "n", .context_index = 1, .min = &minexpr_of_symb_dgeqrf_id_A1_in_n, .max = &maxexpr_of_symb_dgeqrf_id_A1_in_n, .cst_inc = 1, .expr_inc = NULL,  .flags = 0x0};

static inline int affinity_of_dgeqrf_id_A1_in(__dague_dgeqrf_id_A1_in_task_t *this_task,
                     dague_data_ref_t *ref)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

  /* Silent Warnings: should look into predicate to know what variables are usefull */
  (void)__dague_handle;
  (void)m;
  (void)n;
  ref->ddesc = (dague_ddesc_t *)__dague_handle->super.dataA1;
  /* Compute data key */
  ref->key = ref->ddesc->data_key(ref->ddesc, m, n);
  return 1;
}
static inline int initial_data_of_dgeqrf_id_A1_in(__dague_dgeqrf_id_A1_in_task_t *this_task,
                     dague_data_ref_t *refs)
{
    const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
    dague_ddesc_t *__d = NULL;
    int __flow_nb = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;

    /* Silent Warnings: should look into predicate to know what variables are usefull */
    (void)__dague_handle;
  (void)m;
  (void)n;
      /** Flow of A */
    __d = (dague_ddesc_t*)__dague_handle->super.dataA1;
    refs[__flow_nb].ddesc = __d;
    refs[__flow_nb].key = __d->data_key(__d, m, n);
    __flow_nb++;

    return __flow_nb;
}

static dague_data_t *flow_of_dgeqrf_id_A1_in_for_A_dep1_atline_51_direct_access(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *assignments)
{
  dague_ddesc_t *__ddesc;
  const int m = assignments->m.value;
  const int n = assignments->n.value;

  /* Silent Warnings: should look into parameters to know what variables are useful */
  (void)m;
  (void)n;
  __ddesc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  if( __ddesc->myrank == __ddesc->rank_of(__ddesc, m, n) )
    return __ddesc->data_of(__ddesc, m, n);
  return NULL;
}

static const dep_t flow_of_dgeqrf_id_A1_in_for_A_dep1_atline_51 = {
  .cond = NULL,  /*  */
  .ctl_gather_nb = NULL,
  .function_id = -1, /* dgeqrf_id_dataA1 */
  .direct_data = (direct_data_lookup_func_t)&flow_of_dgeqrf_id_A1_in_for_A_dep1_atline_51_direct_access,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_A1_in_for_A,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{
  const int m = locals->m.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (m == n);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52_fct }
};
static const dep_t flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52,  /* (m == n) */
  .ctl_gather_nb = NULL,
  .function_id = 3, /* dgeqrf_id_geqrf_dtsqrt */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsqrt_for_A1,
  .dep_index = 0,
  .dep_datatype_index = 0,
  .belongs_to = &flow_of_dgeqrf_id_A1_in_for_A,
};
static inline int expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53_fct(const __dague_dgeqrf_id_internal_handle_t *__dague_handle, const __dague_dgeqrf_id_A1_in_assignment_t *locals)
{
  const int m = locals->m.value;
  const int n = locals->n.value;

  (void)__dague_handle; (void)locals;
  return (m < n);
}
static const expr_t expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53 = {
  .op = EXPR_OP_INLINE,
  .u_expr = { .inline_func_int32 = (expr_op_int32_inline_func_t)expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53_fct }
};
static const dep_t flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53 = {
  .cond = &expr_of_cond_for_flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53,  /* (m < n) */
  .ctl_gather_nb = NULL,
  .function_id = 5, /* dgeqrf_id_geqrf_dtsmqr */
  .direct_data = (direct_data_lookup_func_t)NULL,
  .flow = &flow_of_dgeqrf_id_geqrf_dtsmqr_for_A1,
  .dep_index = 1,
  .dep_datatype_index = 1,
  .belongs_to = &flow_of_dgeqrf_id_A1_in_for_A,
};
static const dague_flow_t flow_of_dgeqrf_id_A1_in_for_A = {
  .name               = "A",
  .sym_type           = SYM_INOUT,
  .flow_flags         = FLOW_ACCESS_READ|FLOW_HAS_IN_DEPS,
  .flow_index         = 0,
  .flow_datatype_mask = 0x3,
  .dep_in     = { &flow_of_dgeqrf_id_A1_in_for_A_dep1_atline_51 },
  .dep_out    = { &flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52,
 &flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53 }
};

static void
iterate_successors_of_dgeqrf_id_A1_in(dague_execution_unit_t *eu, const __dague_dgeqrf_id_A1_in_task_t *this_task,
               uint32_t action_mask, dague_ontask_function_t *ontask, void *ontask_arg)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_execution_context_t nc;  /* generic placeholder for locals */
  dague_dep_data_description_t data;
  int vpid_dst = -1, rank_src = 0, rank_dst = 0;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)rank_src; (void)rank_dst; (void)__dague_handle; (void)vpid_dst;
  (void)m;  (void)n;
  nc.dague_handle = this_task->dague_handle;
  nc.priority     = this_task->priority;
  nc.chore_id     = 0;
#if defined(DISTRIBUTED)
  rank_src = ((dague_ddesc_t*)__dague_handle->super.dataA1)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, n);
#endif
  if( action_mask & 0x3 ) {  /* Flow of Data A */
    data.data   = this_task->data.A.data_out;
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data.layout = data.arena->opaque_dtt;
    data.count  = 1;
    data.displ  = 0;
  if( action_mask & 0x1 ) {
        if( (m == n) ) {
      __dague_dgeqrf_id_geqrf_dtsqrt_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsqrt_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsqrt.function_id];
        const int geqrf_dtsqrt_k = m;
        if( (geqrf_dtsqrt_k >= (0)) && (geqrf_dtsqrt_k <= (KT)) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsqrt_k;
          const int geqrf_dtsqrt_mmax = dgeqrf_id_geqrf_dtsqrt_inline_c_expr5_line_110(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsqrt_mmax;
          const int geqrf_dtsqrt_m = 0;
          if( (geqrf_dtsqrt_m >= (0)) && (geqrf_dtsqrt_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsqrt_m;
#if defined(DISTRIBUTED)
            rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
              vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.k.value);
            nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsqrt_as_expr_fct(__dague_handle, &ncc->locals);
          RELEASE_DEP_OUTPUT(eu, "A", this_task, "A1", &nc, rank_src, rank_dst, &data);
          if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_A1_in_for_A_dep2_atline_52, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
            }
          }
    }
  }
      data.arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data.layout = data.arena->opaque_dtt;
  if( action_mask & 0x2 ) {
        if( (m < n) ) {
      __dague_dgeqrf_id_geqrf_dtsmqr_task_t* ncc = (__dague_dgeqrf_id_geqrf_dtsmqr_task_t*)&nc;
      nc.function = __dague_handle->super.super.functions_array[dgeqrf_id_geqrf_dtsmqr.function_id];
        const int geqrf_dtsmqr_k = m;
        if( (geqrf_dtsmqr_k >= (0)) && (geqrf_dtsmqr_k <= (dgeqrf_id_geqrf_dtsmqr_inline_c_expr3_line_233(__dague_handle, &ncc->locals))) ) {
          assert(&nc.locals[0].value == &ncc->locals.k.value);
          ncc->locals.k.value = geqrf_dtsmqr_k;
          const int geqrf_dtsmqr_mmax = dgeqrf_id_geqrf_dtsmqr_inline_c_expr2_line_234(__dague_handle, &ncc->locals);
          assert(&nc.locals[1].value == &ncc->locals.mmax.value);
          ncc->locals.mmax.value = geqrf_dtsmqr_mmax;
          const int geqrf_dtsmqr_m = 0;
          if( (geqrf_dtsmqr_m >= (0)) && (geqrf_dtsmqr_m <= (ncc->locals.mmax.value)) ) {
            assert(&nc.locals[2].value == &ncc->locals.m.value);
            ncc->locals.m.value = geqrf_dtsmqr_m;
            const int geqrf_dtsmqr_n = n;
            if( (geqrf_dtsmqr_n >= ((ncc->locals.k.value + 1))) && (geqrf_dtsmqr_n <= ((descA1.nt - 1))) ) {
              assert(&nc.locals[3].value == &ncc->locals.n.value);
              ncc->locals.n.value = geqrf_dtsmqr_n;
#if defined(DISTRIBUTED)
              rank_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->rank_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              if( (NULL != eu) && (rank_dst == eu->virtual_process->dague_context->my_rank) )
#endif /* DISTRIBUTED */
                vpid_dst = ((dague_ddesc_t*)__dague_handle->super.dataA2)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA2, ncc->locals.m.value, ncc->locals.n.value);
              nc.priority = __dague_handle->super.super.priority + priority_of_dgeqrf_id_geqrf_dtsmqr_as_expr_fct(__dague_handle, &ncc->locals);
            RELEASE_DEP_OUTPUT(eu, "A", this_task, "A1", &nc, rank_src, rank_dst, &data);
            if( DAGUE_ITERATE_STOP == ontask(eu, &nc, (const dague_execution_context_t *)this_task, &flow_of_dgeqrf_id_A1_in_for_A_dep3_atline_53, &data, rank_src, rank_dst, vpid_dst, ontask_arg) )
  return;
              }
            }
          }
    }
  }
  }
  (void)data;(void)nc;(void)eu;(void)ontask;(void)ontask_arg;(void)rank_dst;(void)action_mask;
}

static int release_deps_of_dgeqrf_id_A1_in(dague_execution_unit_t *eu, __dague_dgeqrf_id_A1_in_task_t *this_task, uint32_t action_mask, dague_remote_deps_t *deps)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (const __dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
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
    arg.output_entry = data_repo_lookup_entry_and_create( eu, A1_in_repo, __jdf2c_hash_A1_in(__dague_handle, (__dague_dgeqrf_id_A1_in_assignment_t*)(&this_task->locals)) );
    arg.output_entry->generator = (void*)this_task;  /* for AYU */
#if defined(DAGUE_SIM)
    assert(arg.output_entry->sim_exec_date == 0);
    arg.output_entry->sim_exec_date = this_task->sim_exec_date;
#endif
  }
  iterate_successors_of_dgeqrf_id_A1_in(eu, this_task, action_mask, dague_release_dep_fct, &arg);

#if defined(DISTRIBUTED)
  if( (action_mask & DAGUE_ACTION_SEND_REMOTE_DEPS) && (NULL != arg.remote_deps)) {
    dague_remote_dep_activate(eu, (dague_execution_context_t *)this_task, arg.remote_deps, arg.remote_deps->outgoing_mask);
  }
#endif

  if(action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) {
    struct dague_vp_s** vps = eu->virtual_process->dague_context->virtual_processes;
    data_repo_entry_addto_usage_limit(A1_in_repo, arg.output_entry->key, arg.output_usage);
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

static int data_lookup_of_dgeqrf_id_A1_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A1_in_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  assignment_t generic_locals[MAX_PARAM_COUNT];  /* generic task locals */
  int target_device = 0; (void)target_device;
  (void)__dague_handle; (void)generic_locals; (void)context;
  dague_data_copy_t *chunk = NULL;
  data_repo_entry_t *entry = NULL;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  /** Lookup the input data, and store them in the context if any */
  if( NULL == (chunk = this_task->data.A.data_in) ) {  /* flow A */
    entry = NULL;
    chunk = dague_data_get_copy(dataA1(m, n), target_device);
    OBJ_RETAIN(chunk);
      this_task->data.A.data_in   = chunk;   /* flow A */
      this_task->data.A.data_repo = entry;
    }
    /* Now get the local version of the data to be worked on */
    if( NULL == chunk ) {
        dague_abort("A NULL input on a READ flow A1_in:A has been forwarded");
    }
    this_task->data.A.data_out = dague_data_get_copy(chunk->original, target_device);
  /** Generate profiling information */
#if defined(DAGUE_PROF_TRACE)
  this_task->prof_info.desc = (dague_ddesc_t*)__dague_handle->super.dataA1;
  this_task->prof_info.id   = ((dague_ddesc_t*)(__dague_handle->super.dataA1))->data_key((dague_ddesc_t*)__dague_handle->super.dataA1, m, n);
#endif  /* defined(DAGUE_PROF_TRACE) */
  (void)m;  (void)n; (void)chunk; (void)entry;

  return DAGUE_HOOK_RETURN_DONE;
}

static int datatype_lookup_of_dgeqrf_id_A1_in(dague_execution_unit_t *eu, const __dague_dgeqrf_id_A1_in_task_t *this_task,
              uint32_t* flow_mask, dague_dep_data_description_t* data)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)__dague_handle; (void)eu; (void)this_task; (void)data;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
if( (*flow_mask) & 0x80000000U ) { /* these are the input flows */
if( (*flow_mask) & 0x1U ) {  /* Flow A */
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x1U;
      return DAGUE_HOOK_RETURN_NEXT;
}  /* (flow_mask & 0x1U) */
    goto no_mask_match;
  }  /* input flows */
if( (*flow_mask) & 0x3U ) {  /* Flow A */
    if( ((*flow_mask) & 0x3U) && ((m == n)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_UPPER_TILE_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x3U;
      return DAGUE_HOOK_RETURN_NEXT;
    }
    if( ((*flow_mask) & 0x0U) && ((m < n)) ) {
    data->arena  = __dague_handle->super.arenas[DAGUE_dgeqrf_id_DEFAULT_ARENA];
    data->layout = data->arena->opaque_dtt;
    data->count  = 1;
    data->displ  = 0;
      (*flow_mask) &= ~0x0U;
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
  (void)m;  (void)n;
  return DAGUE_HOOK_RETURN_DONE;
}
static int hook_of_dgeqrf_id_A1_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A1_in_task_t *this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
  (void)context; (void)__dague_handle;
  const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
  (void)m;  (void)n;

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

/*-----                                  A1_in BODY                                  -----*/

  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id],
                        this_task);
#line 56 "dgeqrf_id.jdf"
{
    /* Nothing */
}

#line 5477 "dgeqrf_id.c"
/*-----                              END OF A1_in BODY                                -----*/



#endif /*!defined(DAGUE_PROF_DRY_BODY)*/

  return DAGUE_HOOK_RETURN_DONE;
}
static int complete_hook_of_dgeqrf_id_A1_in(dague_execution_unit_t *context, __dague_dgeqrf_id_A1_in_task_t *this_task)
{
  const __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)this_task->dague_handle;
#if defined(DISTRIBUTED)
    const int m = this_task->locals.m.value;
  const int n = this_task->locals.n.value;
#endif  /* defined(DISTRIBUTED) */
  (void)context; (void)__dague_handle;
  DAGUE_TASK_PROF_TRACE(context->eu_profile,
                        this_task->dague_handle->profiling_array[2*this_task->function->function_id+1],
                        this_task);
#if defined(DISTRIBUTED)
  /** If not working on distributed, there is no risk that data is not in place */
  (void)m;  (void)n;

#endif /* DISTRIBUTED */
#if defined(DAGUE_PROF_GRAPHER)
  dague_prof_grapher_task((dague_execution_context_t*)this_task, context->th_id, context->virtual_process->vp_id, __jdf2c_hash_A1_in(__dague_handle, &this_task->locals));
#endif  /* defined(DAGUE_PROF_GRAPHER) */
  release_deps_of_dgeqrf_id_A1_in(context, this_task,
      DAGUE_ACTION_RELEASE_REMOTE_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_DEPS |
      DAGUE_ACTION_RELEASE_LOCAL_REFS |
      0x3,  /* mask of all dep_index */ 
      NULL);
  return DAGUE_HOOK_RETURN_DONE;
}

static int dgeqrf_id_A1_in_internal_init(dague_execution_unit_t * eu, __dague_dgeqrf_id_A1_in_task_t * this_task)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_dependencies_t *dep = NULL;
  __dague_dgeqrf_id_A1_in_assignment_t assignments;
  int32_t  m, n;
  uint32_t nb_tasks = 0;
  int32_t __jdf2c_m_min = 0x7fffffff,__jdf2c_n_min = 0x7fffffff;
  int32_t __jdf2c_m_max = 0,__jdf2c_n_max = 0;
  int32_t __jdf2c_m_start, __jdf2c_m_end, __jdf2c_m_inc;
  int32_t __jdf2c_n_start, __jdf2c_n_end, __jdf2c_n_inc;
  /* First, find the min and max value for each of the dimensions */
  for( assignments.m.value = m = 0;
      assignments.m.value <= KT;
      assignments.m.value += 1, m = assignments.m.value) {
    __jdf2c_m_max = dague_imax(__jdf2c_m_max, assignments.m.value);
    __jdf2c_m_min = dague_imin(__jdf2c_m_min, assignments.m.value);
    for( assignments.n.value = n = m;
        assignments.n.value <= (descA1.nt - 1);
        assignments.n.value += 1, n = assignments.n.value) {
      __jdf2c_n_max = dague_imax(__jdf2c_n_max, assignments.n.value);
      __jdf2c_n_min = dague_imin(__jdf2c_n_min, assignments.n.value);
      if( !A1_in_pred(assignments.m.value, assignments.n.value) ) continue;
      nb_tasks++;
    }
  }
  /* Set the range variables for the collision-free hash-computation */
  __dague_handle->A1_in_m_range = (__jdf2c_m_max - __jdf2c_m_min) + 1;
  __dague_handle->A1_in_n_range = (__jdf2c_n_max - __jdf2c_n_min) + 1;
  DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocating dependencies array for dgeqrf_id_A1_in_internal_init (nb_tasks = %d)", nb_tasks);
  if( 0 != nb_tasks ) {

  /**
   * Now, for each of the dimensions, re-iterate on the space,
   * and if at least one value is defined, allocate arrays to point
   * to it. Array dimensions are defined by the (rough) observation above
   **/
    dep = NULL;
    __jdf2c_m_start = 0;
    __jdf2c_m_end = KT;
    __jdf2c_m_inc = 1;
    for(m = dague_imax(__jdf2c_m_start, __jdf2c_m_min); m <= dague_imin(__jdf2c_m_end, __jdf2c_m_max); m+=__jdf2c_m_inc) {
      assignments.m.value = m;
      __jdf2c_n_start = m;
      __jdf2c_n_end = (descA1.nt - 1);
      __jdf2c_n_inc = 1;
      for(n = dague_imax(__jdf2c_n_start, __jdf2c_n_min); n <= dague_imin(__jdf2c_n_end, __jdf2c_n_max); n+=__jdf2c_n_inc) {
        assignments.n.value = n;
        if( !A1_in_pred(m, n) ) continue;
        /* We did find one! Allocate the dependencies array. */
        if( dep == NULL ) {
          ALLOCATE_DEP_TRACKING(dep, __jdf2c_m_min, __jdf2c_m_max, "m", &symb_dgeqrf_id_A1_in_m, NULL, DAGUE_DEPENDENCIES_FLAG_NEXT);
        }
        if( dep->u.next[m-__jdf2c_m_min] == NULL ) {
          ALLOCATE_DEP_TRACKING(dep->u.next[m-__jdf2c_m_min], __jdf2c_n_min, __jdf2c_n_max, "n", &symb_dgeqrf_id_A1_in_n, dep, DAGUE_DEPENDENCIES_FLAG_FINAL);
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

  AYU_REGISTER_TASK(&dgeqrf_id_A1_in);
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

static int __jdf2c_startup_A1_in(dague_execution_unit_t * eu, __dague_dgeqrf_id_A1_in_task_t *this_task)
{
  __dague_dgeqrf_id_A1_in_task_t* new_task;
  __dague_dgeqrf_id_internal_handle_t* __dague_handle = (__dague_dgeqrf_id_internal_handle_t*)this_task->dague_handle;
  dague_context_t           *context = __dague_handle->super.super.context;
  int vpid = 0, nb_tasks = 0;
  size_t total_nb_tasks = 0;
  dague_list_item_t* pready_ring = NULL;
  int m = this_task->locals.m.value;  /* retrieve value saved during the last iteration */
  int n = this_task->locals.n.value;  /* retrieve value saved during the last iteration */
  if( 0 != this_task->locals.reserved[0].value ) {
    this_task->locals.reserved[0].value = 1; /* reset the submission process */
    goto after_insert_task;
  }
  this_task->locals.reserved[0].value = 1; /* a sane default value */
  for(this_task->locals.m.value = m = 0;
      this_task->locals.m.value <= KT;
      this_task->locals.m.value += 1, m = this_task->locals.m.value) {
    for(this_task->locals.n.value = n = m;
        this_task->locals.n.value <= (descA1.nt - 1);
        this_task->locals.n.value += 1, n = this_task->locals.n.value) {
      if( !A1_in_pred(m, n) ) continue;
      if( NULL != ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of ) {
        vpid = ((dague_ddesc_t*)__dague_handle->super.dataA1)->vpid_of((dague_ddesc_t*)__dague_handle->super.dataA1, m, n);
        assert(context->nb_vp >= vpid);
      }
      new_task = (__dague_dgeqrf_id_A1_in_task_t*)dague_thread_mempool_allocate( context->virtual_processes[0]->execution_units[0]->context_mempool );
      new_task->status = DAGUE_TASK_STATUS_NONE;
      /* Copy only the valid elements from this_task to new_task one */
      new_task->dague_handle = this_task->dague_handle;
      new_task->function     = __dague_handle->super.super.functions_array[dgeqrf_id_A1_in.function_id];
      new_task->chore_id     = 0;
      new_task->locals.m.value = this_task->locals.m.value;
      new_task->locals.n.value = this_task->locals.n.value;
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

static const __dague_chore_t __dgeqrf_id_A1_in_chores[] ={
    { .type     = DAGUE_DEV_CPU,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)hook_of_dgeqrf_id_A1_in },
    { .type     = DAGUE_DEV_NONE,
      .evaluate = NULL,
      .hook     = (dague_hook_t*)NULL },  /* End marker */
};

static const dague_function_t dgeqrf_id_A1_in = {
  .name = "A1_in",
  .function_id = 0,
  .nb_flows = 1,
  .nb_parameters = 2,
  .nb_locals = 2,
  .params = { &symb_dgeqrf_id_A1_in_m, &symb_dgeqrf_id_A1_in_n, NULL },
  .locals = { &symb_dgeqrf_id_A1_in_m, &symb_dgeqrf_id_A1_in_n, NULL },
  .data_affinity = (dague_data_ref_fn_t*)affinity_of_dgeqrf_id_A1_in,
  .initial_data = (dague_data_ref_fn_t*)initial_data_of_dgeqrf_id_A1_in,
  .final_data = (dague_data_ref_fn_t*)NULL,
  .priority = NULL,
  .in = { &flow_of_dgeqrf_id_A1_in_for_A, NULL },
  .out = { &flow_of_dgeqrf_id_A1_in_for_A, NULL },
  .flags = 0x0 | DAGUE_HAS_IN_IN_DEPENDENCIES | DAGUE_USE_DEPS_MASK,
  .dependencies_goal = 0x1,
  .key = (dague_functionkey_fn_t*)__jdf2c_hash_A1_in,
  .fini = (dague_hook_t*)NULL,
  .incarnations = __dgeqrf_id_A1_in_chores,
  .find_deps = dague_default_find_deps,
  .iterate_successors = (dague_traverse_function_t*)iterate_successors_of_dgeqrf_id_A1_in,
  .iterate_predecessors = (dague_traverse_function_t*)NULL,
  .release_deps = (dague_release_deps_t*)release_deps_of_dgeqrf_id_A1_in,
  .prepare_input = (dague_hook_t*)data_lookup_of_dgeqrf_id_A1_in,
  .prepare_output = (dague_hook_t*)NULL,
  .get_datatype = (dague_datatype_lookup_t*)datatype_lookup_of_dgeqrf_id_A1_in,
  .complete_execution = (dague_hook_t*)complete_hook_of_dgeqrf_id_A1_in,
  .release_task = (dague_hook_t*)dague_release_task_to_mempool,
#if defined(DAGUE_SIM)
  .sim_cost_fct = (dague_sim_cost_fct_t*)NULL,
#endif
};


static const dague_function_t *dgeqrf_id_functions[] = {
  &dgeqrf_id_A1_in,
  &dgeqrf_id_A2_in,
  &dgeqrf_id_geqrf_dtsqrt_out_A1,
  &dgeqrf_id_geqrf_dtsqrt,
  &dgeqrf_id_geqrf_dtsmqr_out_A1,
  &dgeqrf_id_geqrf_dtsmqr
};

static void dgeqrf_id_startup(dague_context_t *context, __dague_dgeqrf_id_internal_handle_t *__dague_handle, dague_list_item_t ** ready_tasks)
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
      dague_ddesc = (dague_ddesc_t*)__dague_handle->super.dataT2;
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
static void dgeqrf_id_destructor( __dague_dgeqrf_id_internal_handle_t *handle )
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
   data_repo_destroy_nothreadsafe(handle->repositories[5]);  /* geqrf_dtsmqr */
   data_repo_destroy_nothreadsafe(handle->repositories[3]);  /* geqrf_dtsqrt */
   data_repo_destroy_nothreadsafe(handle->repositories[1]);  /* A2_in */
   data_repo_destroy_nothreadsafe(handle->repositories[0]);  /* A1_in */
  /* Release the dependencies arrays for this object */
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
  dague_ddesc = (dague_ddesc_t*)handle->super.dataT2;
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
#undef dataT2
#undef ib
#undef optid
#undef p_work
#undef p_tau
#undef descA1
#undef descA2
#undef descT2
#undef KT
#undef smallnb

dague_dgeqrf_id_handle_t *dague_dgeqrf_id_new(dague_ddesc_t * dataA1 /* data dataA1 */, dague_ddesc_t * dataA2 /* data dataA2 */, dague_ddesc_t * dataT2 /* data dataT2 */, int ib, int optid, dague_memory_pool_t * p_work, dague_memory_pool_t * p_tau)
{
  __dague_dgeqrf_id_internal_handle_t *__dague_handle = (__dague_dgeqrf_id_internal_handle_t *)calloc(1, sizeof(__dague_dgeqrf_id_internal_handle_t));
  dague_function_t* func;
  int i, j;
  /* Dump the hidden parameters */
  tiled_matrix_desc_t descA1;
  tiled_matrix_desc_t descA2;
  tiled_matrix_desc_t descT2;
  int KT;
  int smallnb;
  __dague_handle->super.super.nb_functions = DAGUE_dgeqrf_id_NB_FUNCTIONS;
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
    memcpy((dague_function_t*)__dague_handle->super.super.functions_array[i], dgeqrf_id_functions[i], sizeof(dague_function_t));
    for( j = 0; NULL != func->incarnations[j].hook; j++);
    func->incarnations = (__dague_chore_t*)malloc((j+1) * sizeof(__dague_chore_t));
    memcpy((__dague_chore_t*)func->incarnations, dgeqrf_id_functions[i]->incarnations, (j+1) * sizeof(__dague_chore_t));

    /* Add a placeholder for initialization and startup task */
    __dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+i] = func = (dague_function_t*)malloc(sizeof(dague_function_t));
    memcpy(func, (void*)&__dague_generic_startup, sizeof(dague_function_t));
    func->function_id = __dague_handle->super.super.nb_functions + i;
    func->incarnations = (__dague_chore_t*)malloc(2 * sizeof(__dague_chore_t));
    memcpy((__dague_chore_t*)func->incarnations, (void*)__dague_generic_startup.incarnations, 2 * sizeof(__dague_chore_t));
  }
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+0];
  func->name = "Generic Startup for geqrf_dtsmqr";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_geqrf_dtsmqr_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+1];
  func->name = "Generic Startup for geqrf_dtsmqr_out_A1";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_geqrf_dtsmqr_out_A1_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+2];
  func->name = "Generic Startup for geqrf_dtsqrt";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_geqrf_dtsqrt_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+3];
  func->name = "Generic Startup for geqrf_dtsqrt_out_A1";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_geqrf_dtsqrt_out_A1_internal_init;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+4];
  func->name = "Generic Startup for A2_in";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_A2_in_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_A2_in;
  func = (dague_function_t *)__dague_handle->super.super.functions_array[__dague_handle->super.super.nb_functions+5];
  func->name = "Generic Startup for A1_in";
  func->prepare_input = (dague_hook_t*)dgeqrf_id_A1_in_internal_init;
  ((__dague_chore_t*)&func->incarnations[0])->hook = (dague_hook_t *)__jdf2c_startup_A1_in;
  /* Compute the number of arenas: */
  /*   DAGUE_dgeqrf_id_DEFAULT_ARENA  ->  0 */
  /*   DAGUE_dgeqrf_id_UPPER_TILE_ARENA  ->  1 */
  /*   DAGUE_dgeqrf_id_LITTLE_T_ARENA  ->  2 */
  __dague_handle->super.arenas_size = 3;
  __dague_handle->super.arenas = (dague_arena_t **)malloc(__dague_handle->super.arenas_size * sizeof(dague_arena_t*));
  for(i = 0; i < __dague_handle->super.arenas_size; i++) {
    __dague_handle->super.arenas[i] = (dague_arena_t*)calloc(1, sizeof(dague_arena_t));
  }
  /* Now the Parameter-dependent structures: */
  __dague_handle->super.dataA1 = dataA1;
  __dague_handle->super.dataA2 = dataA2;
  __dague_handle->super.dataT2 = dataT2;
  __dague_handle->super.ib = ib;
  __dague_handle->super.optid = optid;
  __dague_handle->super.p_work = p_work;
  __dague_handle->super.p_tau = p_tau;
  __dague_handle->super.descA1 = descA1 = *((tiled_matrix_desc_t*)dataA1);
  __dague_handle->super.descA2 = descA2 = *((tiled_matrix_desc_t*)dataA2);
  __dague_handle->super.descT2 = descT2 = *((tiled_matrix_desc_t*)dataT2);
  __dague_handle->super.KT = KT = descA1.nt-1;
  __dague_handle->super.smallnb = smallnb = descA2.nb;
  /* If profiling is enabled, the keys for profiling */
#  if defined(DAGUE_PROF_TRACE)
  __dague_handle->super.super.profiling_array = dgeqrf_id_profiling_array;
  if( -1 == dgeqrf_id_profiling_array[0] ) {
    dague_profiling_add_dictionary_keyword("geqrf_dtsmqr", "fill:CC2828",
                                       sizeof(dague_profile_ddesc_info_t)+4*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};mmax{int32_t};m{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dgeqrf_id_geqrf_dtsmqr.function_id /* geqrf_dtsmqr start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dgeqrf_id_geqrf_dtsmqr.function_id /* geqrf_dtsmqr end key */]);

    dague_profiling_add_dictionary_keyword("geqrf_dtsqrt", "fill:CCCC28",
                                       sizeof(dague_profile_ddesc_info_t)+3*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};k{int32_t};mmax{int32_t};m{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dgeqrf_id_geqrf_dtsqrt.function_id /* geqrf_dtsqrt start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dgeqrf_id_geqrf_dtsqrt.function_id /* geqrf_dtsqrt end key */]);

    dague_profiling_add_dictionary_keyword("A2_in", "fill:28CC28",
                                       sizeof(dague_profile_ddesc_info_t)+3*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};m{int32_t};nmin{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dgeqrf_id_A2_in.function_id /* A2_in start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dgeqrf_id_A2_in.function_id /* A2_in end key */]);

    dague_profiling_add_dictionary_keyword("A1_in", "fill:28CCCC",
                                       sizeof(dague_profile_ddesc_info_t)+2*sizeof(assignment_t),
                                       "ddesc_unique_key{uint64_t};ddesc_data_id{uint32_t};ddessc_padding{uint32_t};m{int32_t};n{int32_t}",
                                       (int*)&__dague_handle->super.super.profiling_array[0 + 2 * dgeqrf_id_A1_in.function_id /* A1_in start key */],
                                       (int*)&__dague_handle->super.super.profiling_array[1 + 2 * dgeqrf_id_A1_in.function_id /* A1_in end key */]);

  }
#  endif /* defined(DAGUE_PROF_TRACE) */
  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsmqr);
  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsmqr_out_A1);
  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsqrt);
  AYU_REGISTER_TASK(&dgeqrf_id_geqrf_dtsqrt_out_A1);
  AYU_REGISTER_TASK(&dgeqrf_id_A2_in);
  AYU_REGISTER_TASK(&dgeqrf_id_A1_in);
  __dague_handle->super.super.repo_array = __dague_handle->repositories;
  __dague_handle->super.super.startup_hook = (dague_startup_fn_t)dgeqrf_id_startup;
  __dague_handle->super.super.destructor   = (dague_destruct_fn_t)dgeqrf_id_destructor;
  (void)dague_handle_reserve_id((dague_handle_t*)__dague_handle);
  return (dague_dgeqrf_id_handle_t*)__dague_handle;
}

