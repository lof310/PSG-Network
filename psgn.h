#ifndef PSGN_H
#define PSGN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ───────────────────────────────────────────────────── */
#define MAX_TOKEN_LEN       64
#define MAX_NODES           8192
#define HASHMAP_CAP         4096
#define MAX_EXEMPLARS       32
#define MAX_SEQ_BUF         256

/* ── SDR Operations ──────────────────────────────────────────────── */
uint8_t *sdr_create(int size);
void sdr_free(uint8_t *s);
void sdr_copy(uint8_t *dst, const uint8_t *src, int size);
void sdr_clear(uint8_t *s, int size);
void sdr_or(uint8_t *dst, const uint8_t *a, const uint8_t *b, int size);
void sdr_or_inplace(uint8_t *dst, const uint8_t *src, int size);
void sdr_and(uint8_t *dst, const uint8_t *a, const uint8_t *b, int size);
void sdr_xor_inplace(uint8_t *dst, const uint8_t *src, int size);
void sdr_xor_shifted_inplace(uint8_t *dst, const uint8_t *src, int size, int shift);
void sdr_or_shifted_inplace(uint8_t *dst, const uint8_t *src, int size, int shift);
int sdr_count(const uint8_t *s, int size);
void sdr_circular_shift(uint8_t *dst, const uint8_t *src, int size, int shift);
void sdr_random_sample(int n, int k, int *out);

/* ── HashMap ─────────────────────────────────────────────────────── */
typedef struct {
    char key[MAX_TOKEN_LEN];
    int value;
    int occupied;
} HashEntry;

typedef struct {
    HashEntry entries[HASHMAP_CAP];
    int count;
} HashMap;

void hashmap_init(HashMap *m);
int hashmap_put(HashMap *m, const char *key, int value);
int hashmap_get(const HashMap *m, const char *key, int *value);

/* ── ScalarEncoder ───────────────────────────────────────────────── */
typedef struct {
    int sdr_size;
    int w;
    double min_val;
    double max_val;
    int num_buckets;
} ScalarEncoder;

void scalar_encoder_init(ScalarEncoder *e, int sdr_size, int w, double min_v, double max_v);
void scalar_encoder_encode(const ScalarEncoder *e, double value, uint8_t *out);

/* ── SDREncoder ──────────────────────────────────────────────────── */
typedef struct {
    int sdr_size;
    double sparsity;
    int num_active_bits;
    ScalarEncoder scalar_enc;
    HashMap token_to_id;
    char id_to_token[MAX_NODES][MAX_TOKEN_LEN];
    uint8_t *token_sdrs[MAX_NODES];
    int *bit_freqs[MAX_NODES];
    int next_node_id;
} SDREncoder;

void sdr_encoder_init(SDREncoder *e, int sdr_size, double sparsity);
void sdr_encoder_free(SDREncoder *e);
uint8_t *sdr_encoder_encode(SDREncoder *e, const char *token);
int sdr_encoder_get_node_id(SDREncoder *e, const char *token);
void sdr_encoder_record_bits(SDREncoder *e, const char *token, const uint8_t *sdr);
void sdr_encoder_mutate(SDREncoder *e, const char *a, const char *b, int bits);

/* ── Graph Structures ────────────────────────────────────────────── */
typedef struct Edge {
    int target;
    float weight;
    uint8_t *context_sdr;
    int *scr_exemplars[MAX_EXEMPLARS];
    int scr_exemplar_counts[MAX_EXEMPLARS];
    int num_exemplars;
    struct Edge *next;
} Edge;

typedef struct {
    Edge *adjacency[MAX_NODES];
    int node_exists[MAX_NODES];
    int ctx_sdr_size;
    int scr_sdr_size;
} PredictiveGraph;

typedef struct {
    int node_id;
    float score;
} Prediction;

void graph_init(PredictiveGraph *g, int ctx_size, int scr_size);
void graph_free(PredictiveGraph *g);
void graph_add_node(PredictiveGraph *g, int id);
float graph_get_edge_weight(const PredictiveGraph *g, int from, int to);
void graph_set_edge_weight(PredictiveGraph *g, int from, int to, float w, const uint8_t *ctx, const uint8_t *scr);
int graph_get_predictions(const PredictiveGraph *g, int cur, const uint8_t *ctx_bias, const uint8_t *scr_bias, Prediction *out, int max_out);
int graph_get_all_nodes(const PredictiveGraph *g, int *out, int max_out);
int graph_get_incoming(const PredictiveGraph *g, int node, int *src_out, float *w_out, int max_out);
float graph_contextual_similarity(const PredictiveGraph *g, int a, int b);

/* ── GraphUpdater ────────────────────────────────────────────────── */
typedef struct {
    PredictiveGraph *graph;
    SDREncoder *encoder;
    float learning_rate;
    float decay_rate;
    float overlap_threshold;
    float neuroplasticity_threshold;
    int drift_rate;
    int neuroplastic_enabled;
    int previous_node_id;
    int update_count;
} GraphUpdater;

void updater_init(GraphUpdater *u, PredictiveGraph *g, SDREncoder *enc, float lr, float dr);
int updater_update(GraphUpdater *u, const uint8_t *sdr, int node_id, const uint8_t *ctx, const uint8_t *scr);

/* ── WorkingMemory ───────────────────────────────────────────────── */
typedef struct {
    int sdr_size;
    uint8_t *buffer_sdr;
    uint8_t *seed_sdr;
    uint8_t *pos_sdrs[16];
    int held_count;
} WorkingMemory;

void wm_init(WorkingMemory *wm, int sdr_size);
void wm_free(WorkingMemory *wm);
void wm_hold(WorkingMemory *wm, const uint8_t *sdr);
uint8_t *wm_read(const WorkingMemory *wm);
void wm_clear(WorkingMemory *wm);

/* ── HierarchicalNetwork ─────────────────────────────────────────── */
typedef struct {
    SDREncoder l1_encoder;
    PredictiveGraph l1_graph;
    GraphUpdater l1_updater;
    SDREncoder l2_encoder;
    PredictiveGraph l2_graph;
    GraphUpdater l2_updater;
    WorkingMemory scratchpad;
    char l1_seq_buf[MAX_SEQ_BUF][MAX_TOKEN_LEN];
    int l1_seq_len;
    uint8_t *current_l2_sdr;
} HierarchicalNetwork;

void hierarchy_init(HierarchicalNetwork *h, int l1_size, double l1_sp, int l2_size, double l2_sp, float decay);
void hierarchy_free(HierarchicalNetwork *h);
void hierarchy_get_dynamic_context(HierarchicalNetwork *h, uint8_t *out);
int hierarchy_process_token(HierarchicalNetwork *h, const char *word, char *concept_out, int *trigger_out, int train);
void hierarchy_reset_sequence(HierarchicalNetwork *h);
const char *hierarchy_get_l2_prediction(HierarchicalNetwork *h);

/* ── PSGN Main Interface ─────────────────────────────────────────── */
typedef struct {
    HierarchicalNetwork hierarchy;
} PSGN;

PSGN *psgn_create(int l1_size, double l1_sp, int l2_size, double l2_sp, float decay);
void psgn_free(PSGN *p);
int psgn_tokenize(const char *text, char ***tokens_out);
void psgn_read_text(PSGN *p, const char *text);
char *psgn_generate_text(PSGN *p, const char *seq, int length, int full);

#ifdef __cplusplus
}
#endif

#endif /* PSGN_H */
