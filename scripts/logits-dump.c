/*
 * Reference logits for the qwen3 conformance check.
 *
 * Loads a GGUF through llama.cpp, decodes one fixed token sequence with no sampling and no chat
 * template, and prints the largest logits of the last position as JSON:
 *
 *   {"arch":"qwen3","n_vocab":151936,"tokens":[....],"top":[[id,logit],...]}
 *
 * The sequence is the token ids passed on the command line, so the same ids can be fed to
 * `vipera`'s backend and the two sets of logits compared. Building it against a llama.cpp tree is
 * what `scripts/logits-dump.sh` does; this file exists so the "reference run" is a run, not a
 * number copied out of a table.
 *
 *   gcc -O2 scripts/logits-dump.c -I<llama.cpp>/include -L<llama.cpp>/build/bin -lllama -o logits-dump
 *   ./logits-dump model.gguf 16 12095 13 3555
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

/* Enough of the top of the distribution to pin the ordering without printing 151936 numbers. */
#define TOP_N 16

/* The top rows so far, highest first: a linear scan beats sorting the whole vocabulary. */
struct top {
    int32_t id[TOP_N];
    float value[TOP_N];
    int filled;
};

static void offer(struct top *top, int32_t id, float value) {
    if (top->filled == TOP_N && value <= top->value[TOP_N - 1]) {
        return;
    }
    int at = top->filled < TOP_N ? top->filled : TOP_N - 1;
    while (at > 0 && top->value[at - 1] < value) {
        top->id[at] = top->id[at - 1];
        top->value[at] = top->value[at - 1];
        at--;
    }
    top->id[at] = id;
    top->value[at] = value;
    if (top->filled < TOP_N) {
        top->filled++;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <token> [token ...]\n", argv[0]);
        return 2;
    }

    const int32_t n_tokens = (int32_t) (argc - 2);
    llama_token *tokens = malloc(sizeof(llama_token) * (size_t) n_tokens);
    for (int32_t at = 0; at < n_tokens; at++) {
        tokens[at] = (llama_token) strtol(argv[at + 2], NULL, 10);
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(argv[1], model_params);
    if (model == NULL) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }

    struct llama_context_params context_params = llama_context_default_params();
    /* Room for the sequence plus the usual slack: a context of only a few positions makes
     * llama.cpp reserve a graph that is too small and hands back a row nobody computed. The room
     * follows the sequence so a long-context reference (thousands of positions) is one decode. */
    const int32_t room = n_tokens + 8 > 512 ? n_tokens + 8 : 512;
    context_params.n_ctx = room;
    context_params.n_batch = room;
    context_params.n_threads = 8;
    context_params.n_threads_batch = 8;
    struct llama_context *context = llama_init_from_model(model, context_params);
    if (context == NULL) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t at = 0; at < n_tokens; at++) {
        batch.token[at] = tokens[at];
        batch.pos[at] = at;
        batch.n_seq_id[at] = 1;
        batch.seq_id[at][0] = 0;
        /* Only the last position is asked for: its logits are the whole comparison. */
        batch.logits[at] = (at == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(context, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const float *logits = llama_get_logits_ith(context, n_tokens - 1);
    if (logits == NULL) {
        fprintf(stderr, "no logits at the last position\n");
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct top top = {0};
    for (int32_t id = 0; id < n_vocab; id++) {
        offer(&top, id, logits[id]);
    }

    char arch[64] = {0};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    printf("{\"arch\":\"%s\",\"n_vocab\":%d,\"tokens\":[", arch, n_vocab);
    for (int32_t at = 0; at < n_tokens; at++) {
        printf("%s%d", at == 0 ? "" : ",", tokens[at]);
    }
    printf("],\"top\":[");
    for (int row = 0; row < top.filled; row++) {
        printf("%s[%d,%.6f]", row == 0 ? "" : ",", top.id[row], top.value[row]);
    }
    printf("]}\n");

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    free(tokens);
    return 0;
}
