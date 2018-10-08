#ifndef _dgeqrf_id_h_
#define _dgeqrf_id_h_
#include "dague.h"
#include "dague/constants.h"
#include "dague/data_distribution.h"
#include "dague/data_internal.h"
#include "dague/debug.h"
#include "dague/ayudame.h"
#include "dague/devices/device.h"
#include <assert.h>

BEGIN_C_DECLS

#define DAGUE_dgeqrf_id_DEFAULT_ARENA    0
#define DAGUE_dgeqrf_id_UPPER_TILE_ARENA    1
#define DAGUE_dgeqrf_id_LITTLE_T_ARENA    2
#define DAGUE_dgeqrf_id_ARENA_INDEX_MIN 3

typedef struct dague_dgeqrf_id_handle_s {
  dague_handle_t super;
#define dgeqrf_id_p_work_SIZE (sizeof(double)*ib*descT2.nb)
#define dgeqrf_id_p_tau_SIZE (sizeof(double)   *descT2.nb)
  /* The list of globals */
  dague_ddesc_t * dataA1 /* data dataA1 */;
  dague_ddesc_t * dataA2 /* data dataA2 */;
  dague_ddesc_t * dataT2 /* data dataT2 */;
  int ib;
  int optid;
  dague_memory_pool_t * p_work;
  dague_memory_pool_t * p_tau;
  tiled_matrix_desc_t descA1;
  tiled_matrix_desc_t descA2;
  tiled_matrix_desc_t descT2;
  int KT;
  int smallnb;
  /* The array of datatypes (DEFAULT,UPPER_TILE,LITTLE_T and co.) */
  dague_arena_t** arenas;
  int arenas_size;
} dague_dgeqrf_id_handle_t;

#define dgeqrf_id_p_work_SIZE (sizeof(double)*ib*descT2.nb)
#define dgeqrf_id_p_tau_SIZE (sizeof(double)   *descT2.nb)
extern dague_dgeqrf_id_handle_t *dague_dgeqrf_id_new(dague_ddesc_t * dataA1 /* data dataA1 */, dague_ddesc_t * dataA2 /* data dataA2 */, dague_ddesc_t * dataT2 /* data dataT2 */, int ib, int optid, dague_memory_pool_t * p_work, dague_memory_pool_t * p_tau);

typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-4];
} __dague_dgeqrf_id_geqrf_dtsmqr_assignment_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t V;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-4];
} __dague_dgeqrf_id_geqrf_dtsmqr_data_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_geqrf_dtsmqr_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_geqrf_dtsmqr_data_s data;
} __dague_dgeqrf_id_geqrf_dtsmqr_task_t;


typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_s {
  assignment_t k;
  assignment_t n;
  assignment_t mmax;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_data_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_data_s data;
} __dague_dgeqrf_id_geqrf_dtsmqr_out_A1_task_t;


typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_id_geqrf_dtsqrt_assignment_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_id_geqrf_dtsqrt_data_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_geqrf_dtsqrt_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_geqrf_dtsqrt_data_s data;
} __dague_dgeqrf_id_geqrf_dtsqrt_task_t;


typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_data_t;

typedef struct __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_data_s data;
} __dague_dgeqrf_id_geqrf_dtsqrt_out_A1_task_t;


typedef struct __dague_dgeqrf_id_A2_in_assignment_s {
  assignment_t m;
  assignment_t nmin;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dgeqrf_id_A2_in_assignment_t;

typedef struct __dague_dgeqrf_id_A2_in_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_id_A2_in_data_t;

typedef struct __dague_dgeqrf_id_A2_in_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_A2_in_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_A2_in_data_s data;
} __dague_dgeqrf_id_A2_in_task_t;


typedef struct __dague_dgeqrf_id_A1_in_assignment_s {
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dgeqrf_id_A1_in_assignment_t;

typedef struct __dague_dgeqrf_id_A1_in_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dgeqrf_id_A1_in_data_t;

typedef struct __dague_dgeqrf_id_A1_in_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dgeqrf_id_A1_in_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dgeqrf_id_A1_in_data_s data;
} __dague_dgeqrf_id_A1_in_task_t;


END_C_DECLS

#endif /* _dgeqrf_id_h_ */ 
#define dgeqrf_id_p_work_SIZE (sizeof(double)*ib*descT2.nb)
#define dgeqrf_id_p_tau_SIZE (sizeof(double)   *descT2.nb)
