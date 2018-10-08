#ifndef _dgeqrf_split_h_
#define _dgeqrf_split_h_
#include "dague.h"
#include "dague/constants.h"
#include "dague/data_distribution.h"
#include "dague/data_internal.h"
#include "dague/debug.h"
#include "dague/ayudame.h"
#include "dague/devices/device.h"
#include <assert.h>

BEGIN_C_DECLS

#define DAGUE_dgeqrf_split_DEFAULT_ARENA    0
#define DAGUE_dgeqrf_split_UPPER_TILE_ARENA    1
#define DAGUE_dgeqrf_split_LOWER_TILE_ARENA    2
#define DAGUE_dgeqrf_split_LITTLE_T_ARENA    3
#define DAGUE_dgeqrf_split_ARENA_INDEX_MIN 4

typedef struct dague_dgeqrf_split_handle_s {
  dague_handle_t super;
#define dgeqrf_split_p_work_SIZE (sizeof(double)*ib*descT1.nb)
#define dgeqrf_split_p_tau_SIZE (sizeof(double)   *descT1.nb)
  /* The list of globals */
  dague_ddesc_t * dataA1 /* data dataA1 */;
  dague_ddesc_t * dataA2 /* data dataA2 */;
  dague_ddesc_t * dataT1 /* data dataT1 */;
  dague_ddesc_t * dataT2 /* data dataT2 */;
  int optqr;
  int optid;
  dague_memory_pool_t * p_work;
  dague_memory_pool_t * p_tau;
  tiled_matrix_desc_t descA1;
  tiled_matrix_desc_t descA2;
  tiled_matrix_desc_t descT1;
  tiled_matrix_desc_t descT2;
  int ib;
  int KT;
  int smallnb;
  /* The array of datatypes (DEFAULT,UPPER_TILE,LOWER_TILE,LITTLE_T and co.) */
  dague_arena_t** arenas;
  int arenas_size;
} dague_dgeqrf_split_handle_t;

#define dgeqrf_split_p_work_SIZE (sizeof(double)*ib*descT1.nb)
#define dgeqrf_split_p_tau_SIZE (sizeof(double)   *descT1.nb)
extern dague_dgeqrf_split_handle_t *dague_dgeqrf_split_new(dague_ddesc_t * dataA1 /* data dataA1 */, dague_ddesc_t * dataA2 /* data dataA2 */, dague_ddesc_t * dataT1 /* data dataT1 */, dague_ddesc_t * dataT2 /* data dataT2 */, int optqr, int optid, dague_memory_pool_t * p_work, dague_memory_pool_t * p_tau);

typedef struct __dague_dgeqrf_split_geqrf2_dtsmqr_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-4];
} __dague_dgeqrf_split_geqrf2_dtsmqr_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf2_dtsmqr_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t V;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-4];
} __dague_dgeqrf_split_geqrf2_dtsmqr_data_t;

typedef struct __dague_dgeqrf_split_geqrf2_dtsmqr_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf2_dtsmqr_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf2_dtsmqr_data_s data;
} __dague_dgeqrf_split_geqrf2_dtsmqr_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_assignment_s {
  assignment_t k;
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf1_dtsmqr_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t V;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-4];
} __dague_dgeqrf_split_geqrf1_dtsmqr_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dtsmqr_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dtsmqr_data_s data;
} __dague_dgeqrf_split_geqrf1_dtsmqr_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_assignment_s {
  assignment_t k;
  assignment_t n;
  assignment_t mmax;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_data_s data;
} __dague_dgeqrf_split_geqrf1_dtsmqr_out_A1_task_t;


typedef struct __dague_dgeqrf_split_geqrf2_dtsqrt_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf2_dtsqrt_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf2_dtsqrt_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf2_dtsqrt_data_t;

typedef struct __dague_dgeqrf_split_geqrf2_dtsqrt_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf2_dtsqrt_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf2_dtsqrt_data_s data;
} __dague_dgeqrf_split_geqrf2_dtsqrt_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_assignment_s {
  assignment_t k;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_split_geqrf1_dtsqrt_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf1_dtsqrt_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dtsqrt_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dtsqrt_data_s data;
} __dague_dgeqrf_split_geqrf1_dtsqrt_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_data_s data;
} __dague_dgeqrf_split_geqrf1_dtsqrt_out_A1_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dormqr_assignment_s {
  assignment_t k;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_split_geqrf1_dormqr_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dormqr_data_s {
  dague_data_pair_t C;
  dague_data_pair_t A;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_geqrf1_dormqr_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dormqr_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dormqr_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dormqr_data_s data;
} __dague_dgeqrf_split_geqrf1_dormqr_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_assignment_s {
  assignment_t k;
  assignment_t reserved[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_geqrf1_dgeqrt_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_data_s {
  dague_data_pair_t A;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_split_geqrf1_dgeqrt_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dgeqrt_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dgeqrt_data_s data;
} __dague_dgeqrf_split_geqrf1_dgeqrt_task_t;


typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_assignment_s {
  assignment_t k;
  assignment_t reserved[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_assignment_t;

typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_data_t;

typedef struct __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_data_s data;
} __dague_dgeqrf_split_geqrf1_dgeqrt_typechange_task_t;


typedef struct __dague_dgeqrf_split_A2_in_assignment_s {
  assignment_t m;
  assignment_t nmin;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_split_A2_in_assignment_t;

typedef struct __dague_dgeqrf_split_A2_in_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_A2_in_data_t;

typedef struct __dague_dgeqrf_split_A2_in_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_A2_in_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_A2_in_data_s data;
} __dague_dgeqrf_split_A2_in_task_t;


typedef struct __dague_dgeqrf_split_A1_in_assignment_s {
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_split_A1_in_assignment_t;

typedef struct __dague_dgeqrf_split_A1_in_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_split_A1_in_data_t;

typedef struct __dague_dgeqrf_split_A1_in_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_split_A1_in_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_split_A1_in_data_s data;
} __dague_dgeqrf_split_A1_in_task_t;


END_C_DECLS

#endif /* _dgeqrf_split_h_ */ 
#define dgeqrf_split_p_work_SIZE (sizeof(double)*ib*descT1.nb)
#define dgeqrf_split_p_tau_SIZE (sizeof(double)   *descT1.nb)
