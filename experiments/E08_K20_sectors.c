/* E08: Extract sector data S00, S10 for K=20.
 *
 * Compile: gcc -O3 -o E08 E08_K20_sectors.c -lm -lpthread
 * Run:     ./E08 [num_threads]    (default: 8)
 */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

typedef long long ll;
typedef unsigned long long ull;

#define K 20
#define D (2*K - 1)

typedef struct {
    ull p_start, p_end;
    int thread_id;
    ll S00, S10, n00, n10;
    ll S00_c0, S00_c1;
    ull total;
} thread_arg_t;

static volatile int progress_done = 0;
static volatile ull progress_p = 0;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void *worker(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    ull lo = 1ULL << (K - 1);
    ull hi = 1ULL << K;
    int sb = K - 2;
    ull D_bit  = 1ULL << (D - 1);
    ull D1_bit = 1ULL << D;

    ll lS00 = 0, lS10 = 0;
    ll ln00 = 0, ln10 = 0;
    ll lS00_c0 = 0, lS00_c1 = 0;
    ull ltotal = 0;

    for (ull p = t->p_start; p < t->p_end; p++) {
        int a = (p >> sb) & 1;

        for (ull q = lo; q < hi; q++) {
            int c = (q >> sb) & 1;
            if (a & c) continue;

            ull prod = p * q;
            if (!(prod & D_bit) || (prod & D1_bit)) continue;

            int carries[D + 1];
            memset(carries, 0, sizeof(carries));
            for (int j = 0; j < D; j++) {
                int cv = 0;
                int i_lo = (j - K + 1 > 0) ? j - K + 1 : 0;
                int i_hi = (j < K - 1) ? j : K - 1;
                for (int i = i_lo; i <= i_hi; i++)
                    cv += ((p >> i) & 1) * ((q >> (j - i)) & 1);
                carries[j + 1] = (cv + carries[j]) >> 1;
            }

            int M = 0;
            for (int j = D; j >= 1; j--) {
                if (carries[j] > 0) { M = j; break; }
            }
            if (M == 0) continue;

            int val = carries[M - 1] - 1;
            int c_D2 = carries[D - 2];

            if (a == 0 && c == 0) {
                lS00 += val; ln00++;
                if (c_D2 == 0) lS00_c0 += val;
                else           lS00_c1 += val;
            } else if (a == 1 && c == 0) {
                lS10 += val; ln10++;
            }
            ltotal++;
        }

        if (t->thread_id == 0)
            progress_p = p;
    }

    t->S00 = lS00; t->S10 = lS10;
    t->n00 = ln00; t->n10 = ln10;
    t->S00_c0 = lS00_c0; t->S00_c1 = lS00_c1;
    t->total = ltotal;
    return NULL;
}

int main(int argc, char **argv) {
    int nthreads = (argc > 1) ? atoi(argv[1]) : 8;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;

    ull lo = 1ULL << (K - 1);
    ull hi = 1ULL << K;
    ull range = hi - lo;

    printf("E08: K=%d sector extraction (%d threads)\n", K, nthreads);
    printf("  lo=%llu hi=%llu D=%d\n", lo, hi, D);
    printf("  Pairs: %llu x %llu = %.3e\n", range, range,
           (double)range * (double)range);
    fflush(stdout);

    thread_arg_t *args = calloc(nthreads, sizeof(thread_arg_t));
    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));

    ull chunk = range / nthreads;
    for (int i = 0; i < nthreads; i++) {
        args[i].thread_id = i;
        args[i].p_start = lo + i * chunk;
        args[i].p_end = (i == nthreads - 1) ? hi : lo + (i + 1) * chunk;
    }

    double t0 = now_sec();

    for (int i = 0; i < nthreads; i++)
        pthread_create(&threads[i], NULL, worker, &args[i]);

    /* Progress monitor in main thread */
    ull t0_p_end = args[0].p_end;
    while (!progress_done) {
        usleep(10000000);  /* 10s */
        ull pp = progress_p;
        if (pp <= lo) continue;
        double elapsed = now_sec() - t0;
        double frac = (double)(pp - lo) / (double)(t0_p_end - lo);
        double eta = (frac > 0.001) ? elapsed / frac - elapsed : 0;
        printf("  thread0: p=%llu (%.2f%%) elapsed=%.0fs ETA~%.0fs\n",
               pp, frac * 100, elapsed, eta);
        fflush(stdout);
        if (frac >= 0.999) break;
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i], NULL);

    double elapsed = now_sec() - t0;

    ll S00 = 0, S10 = 0, n00 = 0, n10 = 0;
    ll S00_c0 = 0, S00_c1 = 0;
    ull total = 0;
    for (int i = 0; i < nthreads; i++) {
        S00 += args[i].S00; S10 += args[i].S10;
        n00 += args[i].n00; n10 += args[i].n10;
        S00_c0 += args[i].S00_c0; S00_c1 += args[i].S00_c1;
        total += args[i].total;
    }

    double R = (S00 != 0) ? (double)S10 / (double)S00 : 0.0;
    double alpha = (S00_c0 != 0) ? (double)S00_c1 / (double)S00_c0 : 0.0;

    printf("\nK=%d COMPLETED in %.1fs (%d threads)\n", K, elapsed, nthreads);
    printf("  total D-odd pairs: %llu\n", total);
    printf("  S00 = %lld  (n00 = %lld)\n", S00, n00);
    printf("  S10 = %lld  (n10 = %lld)\n", S10, n10);
    printf("  S00_c0 = %lld  S00_c1 = %lld\n", S00_c0, S00_c1);
    printf("  R = S10/S00 = %.10f  (target: -pi = -3.1415926536)\n", R);
    printf("  alpha = S00_c1/S00_c0 = %.10f  (target: 1/6 = 0.1667)\n", alpha);
    printf("  R + pi = %+.6e\n", R + 3.14159265358979);
    printf("  EXACT: S00=%lld S10=%lld S00_c0=%lld S00_c1=%lld\n",
           S00, S10, S00_c0, S00_c1);
    fflush(stdout);

    free(args);
    free(threads);
    return 0;
}
