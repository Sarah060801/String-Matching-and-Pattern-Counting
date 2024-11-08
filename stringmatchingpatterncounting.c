#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>


typedef struct {
    double basic_time;
    double parallel_time;
    double simd_time;
    double parallel_simd_time;
    size_t basic_matches;
    size_t parallel_matches;
    size_t simd_matches;
    size_t parallel_simd_matches;
} TimingResults;


void generate_random_sequence(char* sequence, size_t length) {
    const char bases[] = { 'A', 'C', 'G', 'T' };
    for (size_t i = 0; i < length; i++) {
        sequence[i] = bases[rand() % 4];
    }
    sequence[length] = '\0';
}


void computeLPSArray(const char* pattern, size_t M, size_t* lps) {
    size_t length = 0;
    lps[0] = 0;
    size_t i = 1;

    while (i < M) {
        if (pattern[i] == pattern[length]) {
            length++;
            lps[i] = length;
            i++;
        }
        else {
            if (length != 0) {
                length = lps[length - 1];
            }
            else {
                lps[i] = 0;
                i++;
            }
        }
    }
}


size_t KMP_search(const char* text, const char* pattern) {
    size_t matches = 0;
    size_t N = strlen(text);
    size_t M = strlen(pattern);

    size_t* lps = (size_t*)malloc(M * sizeof(size_t));
    if (!lps) {
        fprintf(stderr, "Memory allocation failed for LPS array!\n");
        return 0;
    }

    computeLPSArray(pattern, M, lps);

    size_t i = 0;
    size_t j = 0;
    while (i < N) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        if (j == M) {
            matches++;
            j = lps[j - 1];
        }
        else if (i < N && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            }
            else {
                i++;
            }
        }
    }
    free(lps);
    return matches;
}


size_t KMP_search_parallel(const char* text, const char* pattern) {
    size_t total_matches = 0;
    size_t N = strlen(text);
    size_t M = strlen(pattern);

    size_t* lps = (size_t*)malloc(M * sizeof(size_t));
    if (!lps) {
        fprintf(stderr, "Memory allocation failed for LPS array!\n");
        return 0;
    }

    computeLPSArray(pattern, M, lps);

#pragma omp parallel reduction(+:total_matches)
    {
#pragma omp for
        for (long long start = 0; start < (long long)(N - M + 1); start++) {
            size_t j = 0;
            while (j < M && text[start + j] == pattern[j]) {
                j++;
            }
            if (j == M) {
                total_matches++;
            }
        }
    }

    free(lps);
    return total_matches;
}


size_t KMP_search_SIMD(const char* text, const char* pattern) {
    size_t matches = 0;
    size_t N = strlen(text);
    size_t M = strlen(pattern);

    
    if (M > 16) {
        return KMP_search(text, pattern);
    }

    
    char padded_pattern[16] = { 0 };
    memcpy(padded_pattern, pattern, M);
    __m128i pattern_vector = _mm_loadu_si128((__m128i*)padded_pattern);

    for (size_t i = 0; i <= N - M; i++) {
        __m128i text_vector = _mm_loadu_si128((__m128i*) & text[i]);
        __m128i cmp = _mm_cmpeq_epi8(text_vector, pattern_vector);
        unsigned int mask = _mm_movemask_epi8(cmp);

        
        unsigned int relevant_bits = (1 << M) - 1;
        if ((mask & relevant_bits) == relevant_bits) {
            matches++;
        }
    }
    return matches;
}


size_t KMP_search_parallel_SIMD(const char* text, const char* pattern) {
    size_t total_matches = 0;
    size_t N = strlen(text);
    size_t M = strlen(pattern);

    
    if (M > 16) {
        return KMP_search_parallel(text, pattern);
    }

    
    char padded_pattern[16] = { 0 };
    memcpy(padded_pattern, pattern, M);
    __m128i pattern_vector = _mm_loadu_si128((__m128i*)padded_pattern);

    
    unsigned int relevant_bits = (1 << M) - 1;

#pragma omp parallel reduction(+:total_matches)
    {
#pragma omp for schedule(dynamic, 1024)
        for (long long i = 0; i <= N - M; i += 1) {
            __m128i text_vector = _mm_loadu_si128((__m128i*) & text[i]);
            __m128i cmp = _mm_cmpeq_epi8(text_vector, pattern_vector);
            unsigned int mask = _mm_movemask_epi8(cmp);

            if ((mask & relevant_bits) == relevant_bits) {
                total_matches++;
            }
        }
    }
    return total_matches;
}


void print_results(const TimingResults* results) {
    printf("\033[1;36m=== Performance Results ===\033[0m\n");
    printf("\033[1;33mBasic KMP:\033[0m\n");
    printf("  Time: %.6f seconds\n", results->basic_time);
    printf("  Matches found: %zu\n", results->basic_matches);

    printf("\n\033[1;33mParallel KMP:\033[0m\n");
    printf("  Time: %.6f seconds\n", results->parallel_time);
    printf("  Matches found: %zu\n", results->parallel_matches);

    printf("\n\033[1;33mSIMD KMP:\033[0m\n");
    printf("  Time: %.6f seconds\n", results->simd_time);
    printf("  Matches found: %zu\n", results->simd_matches);

    printf("\n\033[1;33mParallel+SIMD KMP:\033[0m\n");
    printf("  Time: %.6f seconds\n", results->parallel_simd_time);
    printf("  Matches found: %zu\n", results->parallel_simd_matches);

    printf("\n\033[1;36mSpeedup Analysis:\033[0m\n");
    printf("Parallel vs Basic: %.2fx\n", results->basic_time / results->parallel_time);
    printf("SIMD vs Basic: %.2fx\n", results->basic_time / results->simd_time);
    printf("Parallel+SIMD vs Basic: %.2fx\n", results->basic_time / results->parallel_simd_time);
}


void create_ascii_visualization(const TimingResults* results) {
    printf("\n\033[1;36m=== Execution Time Visualization ===\033[0m\n");

    
    double max_time = results->basic_time;
    if (results->parallel_time > max_time) max_time = results->parallel_time;
    if (results->simd_time > max_time) max_time = results->simd_time;
    if (results->parallel_simd_time > max_time) max_time = results->parallel_simd_time;

    
    const int MAX_WIDTH = 40;
    double scale = MAX_WIDTH / max_time;

    
    printf("Basic        |");
    for (int i = 0; i < (int)(results->basic_time * scale); i++)
        printf("\033[1;31m#\033[0m");
    printf(" %.6fs\n", results->basic_time);

    
    printf("Parallel     |");
    for (int i = 0; i < (int)(results->parallel_time * scale); i++)
        printf("\033[1;32m#\033[0m");
    printf(" %.6fs\n", results->parallel_time);

    
    printf("SIMD         |");
    for (int i = 0; i < (int)(results->simd_time * scale); i++)
        printf("\033[1;34m#\033[0m");
    printf(" %.6fs\n", results->simd_time);

    
    printf("Parallel+SIMD|");
    for (int i = 0; i < (int)(results->parallel_simd_time * scale); i++)
        printf("\033[1;35m#\033[0m");
    printf(" %.6fs\n", results->parallel_simd_time);
}

int main() {
    
    srand((unsigned int)time(NULL));

    
    const size_t text_length = 1000000;  
    const size_t pattern_length = 8;     

    
    char* text = (char*)malloc((text_length + 1) * sizeof(char));
    char* pattern = (char*)malloc((pattern_length + 1) * sizeof(char));

    if (!text || !pattern) {
        fprintf(stderr, "Memory allocation failed!\n");
        if (text) free(text);
        if (pattern) free(pattern);
        return 1;
    }

    
    generate_random_sequence(text, text_length);
    generate_random_sequence(pattern, pattern_length);

    
    printf("\033[1;36mTest Parameters:\033[0m\n");
    printf("Text length: %zu\n", text_length);
    printf("Pattern length: %zu\n", pattern_length);
    printf("Number of OpenMP threads: %d\n", omp_get_max_threads());

    
    TimingResults results = { 0 };

    
    printf("\nRunning basic KMP...\n");
    double start = omp_get_wtime();
    results.basic_matches = KMP_search(text, pattern);
    results.basic_time = omp_get_wtime() - start;

    
    printf("Running parallel KMP...\n");
    start = omp_get_wtime();
    results.parallel_matches = KMP_search_parallel(text, pattern);
    results.parallel_time = omp_get_wtime() - start;

    
    printf("Running SIMD KMP...\n");
    start = omp_get_wtime();
    results.simd_matches = KMP_search_SIMD(text, pattern);
    results.simd_time = omp_get_wtime() - start;

    
    printf("Running Parallel+SIMD KMP...\n");
    start = omp_get_wtime();
    results.parallel_simd_matches = KMP_search_parallel_SIMD(text, pattern);
    results.parallel_simd_time = omp_get_wtime() - start;

    
    print_results(&results);
    create_ascii_visualization(&results);

    
    free(text);
    free(pattern);

    return 0;
}
