/* E45: High-K sector enumeration WITH per-position cascade profiles.
 *
 * Extension of E44: in addition to R(K) = S10/S00, accumulates the
 * per-position cascade sums cas_00[d] and cas_10[d] and the stopping
 * counts stop_00[d] and stop_10[d] for d = 1..D.
 *
 * These profiles enable sine-basis projection w_n^{ab} = <sigma_ab, phi_n>
 * and testing of the Fundamental Mode Dominance hypothesis at K > 15.
 *
 * Compile: cc -O3 -march=native -o E45 E45_high_K_profiles.c -lpthread -lm
 * Run:     ./E45 K_start K_end num_threads
 * Example: ./E45 16 21 8 > E45_profiles.txt 2>E45_progress.log
 *
 * Output: R(K), S00, S10, plus per-position arrays cas_00[d], cas_10[d],
 *         stop_00[d], stop_10[d] for d=1..D.
 *
 * Max K: 32 (products fit in uint64_t for K <= 32).
 */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

typedef long long ll;
typedef unsigned long long ull;

#define KMAX 32
#define DMAX (2 * KMAX - 1)

typedef struct {
    int thread_id;
    int K;
    int nthreads;
    ll S00, S10, n00, n10;
    ll S00_c0, S00_c1;
    ull total;
    ull p_count;
    ll cas_00[DMAX + 2];
    ll cas_10[DMAX + 2];
    ll stop_00[DMAX + 2];
    ll stop_10[DMAX + 2];
} thread_arg_t;

static volatile ull g_progress_count = 0;
static volatile int g_thread0_done = 0;

static double now_sec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void *worker(void *arg)
{
    thread_arg_t *t = (thread_arg_t *)arg;
    const int K  = t->K;
    const int D  = 2 * K - 1;
    const ull lo = 1ULL << (K - 1);
    const ull hi = 1ULL << K;
    const ull b0_hi = lo + (1ULL << (K - 2));
    const int sb = K - 2;
    const ull D_bit  = 1ULL << (D - 1);
    const ull D1_bit = 1ULL << D;
    const int stride = t->nthreads;

    ll lS00 = 0, lS10 = 0;
    ll ln00 = 0, ln10 = 0;
    ll lS00_c0 = 0, lS00_c1 = 0;
    ull ltotal = 0;
    ull lp_count = 0;

    ll lcas_00[DMAX + 2];
    ll lcas_10[DMAX + 2];
    ll lstop_00[DMAX + 2];
    ll lstop_10[DMAX + 2];
    memset(lcas_00,  0, sizeof(lcas_00));
    memset(lcas_10,  0, sizeof(lcas_10));
    memset(lstop_00, 0, sizeof(lstop_00));
    memset(lstop_10, 0, sizeof(lstop_10));

    int carries[DMAX + 2];

    for (ull p = lo + (ull)t->thread_id; p < hi; p += stride) {
        int a = (p >> sb) & 1;

        for (ull q = lo; q < b0_hi; q++) {
            ull prod = p * q;
            if (!(prod & D_bit) || (prod & D1_bit)) continue;

            memset(carries, 0, (D + 1) * sizeof(int));
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

            int val  = carries[M - 1] - 1;
            int c_D2 = carries[D - 2];

            if (a == 0) {
                lS00 += val; ln00++;
                if (c_D2 == 0) lS00_c0 += val;
                else           lS00_c1 += val;
                lcas_00[M] += val;
                lstop_00[M]++;
            } else {
                lS10 += val; ln10++;
                lcas_10[M] += val;
                lstop_10[M]++;
            }
            ltotal++;
        }

        lp_count++;
        if (t->thread_id == 0)
            g_progress_count = lp_count;
    }

    if (t->thread_id == 0)
        g_thread0_done = 1;

    t->S00 = lS00; t->S10 = lS10;
    t->n00 = ln00; t->n10 = ln10;
    t->S00_c0 = lS00_c0; t->S00_c1 = lS00_c1;
    t->total = ltotal;
    t->p_count = lp_count;
    memcpy(t->cas_00,  lcas_00,  (D + 2) * sizeof(ll));
    memcpy(t->cas_10,  lcas_10,  (D + 2) * sizeof(ll));
    memcpy(t->stop_00, lstop_00, (D + 2) * sizeof(ll));
    memcpy(t->stop_10, lstop_10, (D + 2) * sizeof(ll));
    return NULL;
}

static void run_one_K(int K, int nthreads)
{
    const int D   = 2 * K - 1;
    const ull lo  = 1ULL << (K - 1);
    const ull hi  = 1ULL << K;
    const ull range = hi - lo;
    const ull t0_total_p = (range + nthreads - 1) / nthreads;

    printf("\n>>> STARTED K=%d  D=%d  pairs=%.3e  (strided, %d threads)\n",
           K, D, (double)range * (double)(range / 2), nthreads);
    fflush(stdout);
    fprintf(stderr, "\n[K=%d] Starting: %.3e pairs, %d threads (strided)\n",
            K, (double)range * (double)(range / 2), nthreads);

    thread_arg_t *args = calloc(nthreads, sizeof(thread_arg_t));
    pthread_t    *tids = malloc(nthreads * sizeof(pthread_t));

    for (int i = 0; i < nthreads; i++) {
        args[i].thread_id = i;
        args[i].K = K;
        args[i].nthreads = nthreads;
    }

    g_progress_count = 0;
    g_thread0_done = 0;
    double t0 = now_sec();

    for (int i = 0; i < nthreads; i++)
        pthread_create(&tids[i], NULL, worker, &args[i]);

    double last_report = t0;
    while (!g_thread0_done) {
        usleep(1000000);
        if (g_thread0_done) break;
        double now = now_sec();
        if (now - last_report < 30.0) continue;
        last_report = now;
        ull cnt = g_progress_count;
        if (cnt == 0) continue;
        double elapsed = now - t0;
        double frac = (double)cnt / (double)t0_total_p;
        double eta = (frac > 0.001) ? elapsed / frac - elapsed : 0;
        fprintf(stderr, "[K=%d] %.1f%%  elapsed=%.0fs  ETA~%.0fs  (%.1fh/%.1fh)\n",
                K, frac * 100, elapsed, eta,
                elapsed / 3600.0, (elapsed + eta) / 3600.0);
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);

    double elapsed = now_sec() - t0;

    ll  S00 = 0, S10 = 0, n00 = 0, n10 = 0;
    ll  S00_c0 = 0, S00_c1 = 0;
    ull total = 0;
    ll  cas_00[DMAX + 2], cas_10[DMAX + 2];
    ll  stop_00[DMAX + 2], stop_10[DMAX + 2];
    memset(cas_00,  0, sizeof(cas_00));
    memset(cas_10,  0, sizeof(cas_10));
    memset(stop_00, 0, sizeof(stop_00));
    memset(stop_10, 0, sizeof(stop_10));

    for (int i = 0; i < nthreads; i++) {
        S00    += args[i].S00;    S10    += args[i].S10;
        n00    += args[i].n00;    n10    += args[i].n10;
        S00_c0 += args[i].S00_c0; S00_c1 += args[i].S00_c1;
        total  += args[i].total;
        for (int d = 0; d <= D + 1; d++) {
            cas_00[d]  += args[i].cas_00[d];
            cas_10[d]  += args[i].cas_10[d];
            stop_00[d] += args[i].stop_00[d];
            stop_10[d] += args[i].stop_10[d];
        }
    }

    double R     = (S00 != 0)    ? (double)S10    / (double)S00    : 0.0;
    double alpha = (S00_c0 != 0) ? (double)S00_c1 / (double)S00_c0 : 0.0;

    printf(">>> COMPLETED K=%d in %.1fs (%d threads)\n", K, elapsed, nthreads);
    printf("  D        = %d\n", D);
    printf("  D-odd    = %llu\n", total);
    printf("  S00      = %lld    (n00 = %lld)\n", S00, n00);
    printf("  S10      = %lld    (n10 = %lld)\n", S10, n10);
    printf("  S00_c0   = %lld\n", S00_c0);
    printf("  S00_c1   = %lld\n", S00_c1);
    printf("  R        = %.12f\n", R);
    printf("  R + pi   = %+.8e\n", R + M_PI);
    printf("  alpha    = %.12f\n", alpha);
    printf("  |R+pi|   = %.6e   sig.digits = %.2f\n",
           fabs(R + M_PI),
           fabs(R + M_PI) > 0 ? -log10(fabs(R + M_PI)) : 99.0);

    printf("  EXACT K=%d S00_c0=%lld S00_c1=%lld S10=%lld"
           " n00=%lld n10=%lld\n",
           K, S00_c0, S00_c1, S10, n00, n10);

    printf("\n  --- Per-position cascade profiles (d=1..%d) ---\n", D);
    printf("  %4s  %14s  %14s  %10s  %10s\n",
           "d", "cas_00[d]", "cas_10[d]", "stop_00[d]", "stop_10[d]");
    for (int d = 1; d <= D; d++) {
        if (stop_00[d] == 0 && stop_10[d] == 0) continue;
        printf("  %4d  %14lld  %14lld  %10lld  %10lld\n",
               d, cas_00[d], cas_10[d], stop_00[d], stop_10[d]);
    }
    printf("\n");
    fflush(stdout);

    fprintf(stderr, "[K=%d] DONE %.1fs (%.1fh)  R=%.8f  |R+pi|=%.2e\n",
            K, elapsed, elapsed / 3600.0, R, fabs(R + M_PI));

    free(args);
    free(tids);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "E45: High-K sector profiles (extends E44 with per-position data)\n"
            "Usage: %s K_start K_end num_threads\n"
            "  K range: 3..%d (inclusive)\n"
            "  Writes results to stdout, progress to stderr.\n"
            "Example: %s 16 21 8 > E45_profiles.txt 2>E45_progress.log\n",
            argv[0], KMAX, argv[0]);
        return 1;
    }

    int K_start  = atoi(argv[1]);
    int K_end    = atoi(argv[2]);
    int nthreads = atoi(argv[3]);

    if (K_start < 3)      K_start = 3;
    if (K_end   > KMAX)   K_end   = KMAX;
    if (nthreads < 1)     nthreads = 1;
    if (nthreads > 128)   nthreads = 128;

    printf("E45: HIGH-K SECTOR PROFILES (per-position cascade data)\n");
    printf("  K range : %d .. %d\n", K_start, K_end);
    printf("  Threads : %d\n", nthreads);
    printf("  Target R     = -pi  = %.15f\n", -M_PI);
    printf("============================================================\n");
    fflush(stdout);

    for (int K = K_start; K <= K_end; K++)
        run_one_K(K, nthreads);

    printf("\n============================================================\n");
    printf("E45: ALL DONE (K=%d..%d)\n", K_start, K_end);
    fflush(stdout);
    return 0;
}
