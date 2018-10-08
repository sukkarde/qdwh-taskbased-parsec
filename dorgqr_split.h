#ifndef _dorgqr_split_h_
#define _dorgqr_split_h_
#include "dague.h"
#include "dague/constants.h"
#include "dague/data_distribution.h"
#include "dague/data_internal.h"
#include "dague/debug.h"
#include "dague/ayudame.h"
#include "dague/devices/device.h"
#include <assert.h>

BEGIN_C_DECLS

#define DAGUE_dorgqr_split_DEFAULT_ARENA    0
#define DAGUE_dorgqr_split_LOWER_TILE_ARENA    1
#define DAGUE_dorgqr_split_LITTLE_T_ARENA    2
#define DAGUE_dorgqr_split_ARENA_INDEX_MIN 3

typedef struct dague_dorgqr_split_handle_s {
  dague_handle_t super;
#define dorgqr_split_p_work_SIZE ((sizeof(double))*ib)*descT2.nb
  /* The list of globals */
  dague_ddesc_t * dataA1 /* data dataA1 */;
  dague_ddesc_t * dataA2 /* data dataA2 */;
  dague_ddesc_t * dataT1 /* data dataT1 */;
  dague_ddesc_t * dataT2 /* data dataT2 */;
  dague_ddesc_t * dataQ1 /* data dataQ1 */;
  dague_ddesc_t * dataQ2 /* data dataQ2 */;
  int optid;
  dague_memory_pool_t * p_work;
  tiled_matrix_desc_t descA1;
  tiled_matrix_desc_t descA2;
  tiled_matrix_desc_t descT1;
  tiled_matrix_desc_t descT2;
  tiled_matrix_desc_t descQ1;
  tiled_matrix_desc_t descQ2;
  int ib;
  int KT;
  int KT2;
  /* The array of datatypes (DEFAULT,LOWER_TILE,LITTLE_T and co.) */
  dague_arena_t** arenas;
  int arenas_size;
} dague_dorgqr_split_handle_t;

#define dorgqr_split_p_work_SIZE ((sizeof(double))*ib)*descT2.nb
extern dague_dorgqr_split_handle_t *dague_dorgqr_split_new(dague_ddesc_t * dataA1 /* data dataA1 */, dague_ddesc_t * dataA2 /* data dataA2 */, dague_ddesc_t * dataT1 /* data dataT1 */, dague_ddesc_t * dataT2 /* data dataT2 */, dague_ddesc_t * dataQ1 /* data dataQ1 */, dague_ddesc_t * dataQ2 /* data dataQ2 */, int optid, dague_memory_pool_t * p_work);

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_data_s {
  dague_data_pair_t V;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr2_in_A2_task_t;


typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_data_s {
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr2_in_T2_task_t;


typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_assignment_s {
  assignment_t k;
  assignment_t mmax;
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-4];
} __dague_dorgqr_split_ungqr_dtsmqr2_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t V;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-4];
} __dague_dorgqr_split_ungqr_dtsmqr2_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr2_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr2_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr2_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr2_task_t;


typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_s {
  assignment_t k;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_A1_data_s {
  dague_data_pair_t V;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dtsmqr_in_A1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr_in_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr_in_A1_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr_in_A1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_s {
  assignment_t k;
  assignment_t m;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_T1_data_s {
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dtsmqr_in_T1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr_in_T1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr_in_T1_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr_in_T1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dtsmqr1_assignment_s {
  assignment_t k;
  assignment_t m;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-3];
} __dague_dorgqr_split_ungqr_dtsmqr1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr1_data_s {
  dague_data_pair_t A1;
  dague_data_pair_t A2;
  dague_data_pair_t V;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-4];
} __dague_dorgqr_split_ungqr_dtsmqr1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dtsmqr1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dtsmqr1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dtsmqr1_data_s data;
} __dague_dorgqr_split_ungqr_dtsmqr1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_s {
  assignment_t k;
  assignment_t reserved[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr_in_A1_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dormqr_in_A1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr_in_A1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dormqr_in_A1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dormqr_in_A1_data_s data;
} __dague_dorgqr_split_ungqr_dormqr_in_A1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_s {
  assignment_t k;
  assignment_t reserved[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr_in_T1_data_s {
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dormqr_in_T1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr_in_T1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dormqr_in_T1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dormqr_in_T1_data_s data;
} __dague_dorgqr_split_ungqr_dormqr_in_T1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dormqr1_assignment_s {
  assignment_t k;
  assignment_t n;
  assignment_t reserved[MAX_LOCAL_COUNT-2];
} __dague_dorgqr_split_ungqr_dormqr1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr1_data_s {
  dague_data_pair_t C;
  dague_data_pair_t A;
  dague_data_pair_t T;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-3];
} __dague_dorgqr_split_ungqr_dormqr1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dormqr1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dormqr1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dormqr1_data_s data;
} __dague_dorgqr_split_ungqr_dormqr1_task_t;


typedef struct __dague_dorgqr_split_ungqr_dlaset2_assignment_s {
  assignment_t m;
  assignment_t n;
  assignment_t k;
  assignment_t mmax;
  assignment_t reserved[MAX_LOCAL_COUNT-4];
} __dague_dorgqr_split_ungqr_dlaset2_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dlaset2_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dlaset2_data_t;

typedef struct __dague_dorgqr_split_ungqr_dlaset2_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dlaset2_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dlaset2_data_s data;
} __dague_dorgqr_split_ungqr_dlaset2_task_t;


typedef struct __dague_dorgqr_split_ungqr_dlaset1_assignment_s {
  assignment_t m;
  assignment_t n;
  assignment_t k;
  assignment_t mmax;
  assignment_t cq;
  assignment_t reserved[MAX_LOCAL_COUNT-5];
} __dague_dorgqr_split_ungqr_dlaset1_assignment_t;

typedef struct __dague_dorgqr_split_ungqr_dlaset1_data_s {
  dague_data_pair_t A;
  dague_data_pair_t unused[MAX_LOCAL_COUNT-1];
} __dague_dorgqr_split_ungqr_dlaset1_data_t;

typedef struct __dague_dorgqr_split_ungqr_dlaset1_task_s {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_PROF_TRACE)
    dague_profile_ddesc_info_t prof_info;
#endif /* defined(DAGUE_PROF_TRACE) */
    struct __dague_dorgqr_split_ungqr_dlaset1_assignment_s locals;
#if defined(PINS_ENABLE)
    int                        creator_core;
    int                        victim_core;
#endif /* defined(PINS_ENABLE) */
#if defined(DAGUE_SIM)
    int                        sim_exec_date;
#endif
    struct __dague_dorgqr_split_ungqr_dlaset1_data_s data;
} __dague_dorgqr_split_ungqr_dlaset1_task_t;


END_C_DECLS

#endif /* _dorgqr_split_h_ */ 
#define dorgqr_split_p_work_SIZE ((sizeof(double))*ib)*descT2.nb
