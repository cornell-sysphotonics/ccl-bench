
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <nvml.h>

// Number of NVLINKs wired per GPU (A100 has 12)
#define MAX_NVLINKS_PER_GPU 12
#define FIELDS_PER_LINK 2  // TX and RX

// File format constants
#define FILE_MAGIC "NVF1"
#define FILE_VERSION 1

// Queue settings
#define DEFAULT_QUEUE_DEPTH 8192


// Matches Python header structure
typedef struct __attribute__((packed)) {
    char magic[4];
    uint16_t version;
    uint16_t field_size;
    int32_t host_ts_size;
} FileHeader;

// Records
typedef struct {
    uint64_t host_ts;
    nvmlFieldValue_t field_value;
} Record;

// Thread-safe queue
typedef struct {
    Record* buffer;
    int capacity;
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
} RecordQueue;

// Global state
typedef struct {
    volatile sig_atomic_t stop_flag;
    RecordQueue queue;
    char* output_path;
    int gpu_index;
    double interval_ms;
    int nvlinks_to_poll;
    int quiet;  // Suppress output
} AppState;


static AppState g_state = {0};
// Thread-safe queue
int queue_init(RecordQueue* q, int capacity) {
    q->buffer = malloc(capacity * sizeof(Record));
    if (!q->buffer) return -1;
    q->capacity = capacity;
    q->head = q->tail = q->count = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    return 0;
}

void queue_destroy(RecordQueue* q) {
    free(q->buffer);
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
}

int queue_push(RecordQueue* q, const Record* rec) {
    pthread_mutex_lock(&q->mutex);
    
    // Wait with timeout while full
    while (q->count >= q->capacity && !g_state.stop_flag) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50000000; // 50ms timeout
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&q->not_full, &q->mutex, &ts);
    }
    
    if (g_state.stop_flag && q->count >= q->capacity) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    
    q->buffer[q->tail] = *rec;
    q->tail = (q->tail + 1) % q->capacity;
    q->count++;
    
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int queue_pop(RecordQueue* q, Record* rec) {
    pthread_mutex_lock(&q->mutex);
    
    // Wait with timeout while empty
    while (q->count == 0 && !g_state.stop_flag) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50000000; // 50ms timeout
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&q->not_empty, &q->mutex, &ts);
    }
    
    if (q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    
    *rec = q->buffer[q->head];
    q->head = (q->head + 1) % q->capacity;
    q->count--;
    
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}


// Write thread worker
void* writer_worker(void* arg) {
    (void)arg;
    
    FILE* fp = fopen(g_state.output_path, "wb");
    if (!fp) {
        perror("Failed to open output file");
        return NULL;
    }
    
    // Set large buffer for efficiency
    setvbuf(fp, NULL, _IOFBF, 1024 * 1024);
    
    // Write header
    FileHeader header = {
        .magic = {'N', 'V', 'F', '1'},
        .version = FILE_VERSION,
        .field_size = sizeof(nvmlFieldValue_t),
        .host_ts_size = sizeof(uint64_t)
    };
    fwrite(&header, sizeof(header), 1, fp);
    
    // Write records until stopped
    Record rec;
    while (!g_state.stop_flag || g_state.queue.count > 0) {
        if (queue_pop(&g_state.queue, &rec) == 0) {
            fwrite(&rec.host_ts, sizeof(rec.host_ts), 1, fp);
            fwrite(&rec.field_value, sizeof(rec.field_value), 1, fp);
        }
    }
    
    fclose(fp);
    return NULL;
}


// Actual polling loop
void poll_loop(nvmlDevice_t device) {
    // Build field requests for NVLinks (TX and RX)
    int total_fields = g_state.nvlinks_to_poll * FIELDS_PER_LINK;
    nvmlFieldValue_t values[MAX_NVLINKS_PER_GPU * FIELDS_PER_LINK];
    int idx = 0;
    for (int link = 0; link < g_state.nvlinks_to_poll; link++) {
        values[idx].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
        values[idx].scopeId = link;
        idx++;
        values[idx].fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
        values[idx].scopeId = link;
        idx++;
    }
    
    double interval_s = g_state.interval_ms / 1000.0;
    struct timespec next_tick;
    clock_gettime(CLOCK_MONOTONIC, &next_tick);
    
    // Timing diagnostics
    uint64_t last_ts = 0;
    uint64_t poll_count = 0;
    uint64_t total_interval_ns = 0;
    uint64_t min_interval_ns = UINT64_MAX;
    uint64_t max_interval_ns = 0;
    
    // Breakdown timing
    uint64_t total_nvml_ns = 0;
    uint64_t total_queue_ns = 0;
    uint64_t min_nvml_ns = UINT64_MAX;
    
    while (!g_state.stop_flag) {
        // Advance next tick
        long ns = next_tick.tv_nsec + (long)(interval_s * 1e9);
        next_tick.tv_sec += ns / 1000000000;
        next_tick.tv_nsec = ns % 1000000000;
        
        // Sleep until next tick
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_tick, NULL);
        
        if (g_state.stop_flag) break;
        
        // Time the NVML call
        uint64_t before_nvml = get_time_ns();
        nvmlReturn_t ret = nvmlDeviceGetFieldValues(device, total_fields, values);
        uint64_t after_nvml = get_time_ns();
        uint64_t nvml_duration = after_nvml - before_nvml;
        total_nvml_ns += nvml_duration;
        if (nvml_duration < min_nvml_ns) min_nvml_ns = nvml_duration;
        
        uint64_t host_ts = after_nvml;
        
        // Track actual intervals
        if (last_ts != 0) {
            uint64_t actual_interval = host_ts - last_ts;
            total_interval_ns += actual_interval;
            if (actual_interval < min_interval_ns) min_interval_ns = actual_interval;
            if (actual_interval > max_interval_ns) max_interval_ns = actual_interval;
            poll_count++;
            
            // Print every 1000 polls (unless quiet)
            if (!g_state.quiet && poll_count % 1000 == 0) {
                double avg_us = (total_interval_ns / (double)poll_count) / 1000.0;
                double avg_nvml_us = (total_nvml_ns / (double)poll_count) / 1000.0;
                double avg_queue_us = (total_queue_ns / (double)poll_count) / 1000.0;
                fprintf(stderr, "[%lu polls] interval: avg=%.1f us, min=%.1f us | nvml: avg=%.1f us, min=%.1f us | queue: %.1f us\n",
                        poll_count, avg_us, min_interval_ns / 1000.0,
                        avg_nvml_us, min_nvml_ns / 1000.0,
                        avg_queue_us);
            }
        }
        last_ts = host_ts;
        
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "nvmlDeviceGetFieldValues failed: %s\n", 
                    nvmlErrorString(ret));
            continue;
        }
        
        // Time the queue operations
        uint64_t before_queue = get_time_ns();
        for (int i = 0; i < total_fields; i++) {
            Record rec = {
                .host_ts = host_ts,
                .field_value = values[i]
            };
            queue_push(&g_state.queue, &rec);
        }
        total_queue_ns += get_time_ns() - before_queue;
    }
    
    // Final stats on exit (unless quiet)
    if (!g_state.quiet && poll_count > 0) {
        double avg_us = (total_interval_ns / (double)poll_count) / 1000.0;
        double avg_nvml_us = (total_nvml_ns / (double)poll_count) / 1000.0;
        double avg_queue_us = (total_queue_ns / (double)poll_count) / 1000.0;
        fprintf(stderr, "\nFinal stats: %lu polls\n", poll_count);
        fprintf(stderr, "  Interval: avg=%.1f us, min=%.1f us, max=%.1f us\n",
                avg_us, min_interval_ns / 1000.0, max_interval_ns / 1000.0);
        fprintf(stderr, "  NVML call: avg=%.1f us, min=%.1f us\n",
                avg_nvml_us, min_nvml_ns / 1000.0);
        fprintf(stderr, "  Queue ops: avg=%.1f us\n", avg_queue_us);
    }
}

void signal_handler(int signum) {
    (void)signum;  // Suppress unused warning
    g_state.stop_flag = 1;
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --gpu <n>          GPU index to poll (default: 0)\n");
    printf("  --interval-ms <f>  Polling period in ms (default: 1.0)\n");
    printf("  --out <path>       Output file (default: nvlink_trace.bin)\n");
    printf("  --queue-depth <n>  Queue depth (default: %d)\n", DEFAULT_QUEUE_DEPTH);
    printf("  --nvlinks <n>      Number of NVLinks to poll, 1-%d (default: %d)\n", 
           MAX_NVLINKS_PER_GPU, MAX_NVLINKS_PER_GPU);
    printf("  --quiet, -q        Suppress output\n");
}

int main(int argc, char* argv[]) {
    // Defaults
    g_state.gpu_index = 0;
    g_state.interval_ms = 1.0;
    g_state.output_path = "nvlink_trace.bin";
    g_state.nvlinks_to_poll = MAX_NVLINKS_PER_GPU;
    int queue_depth = DEFAULT_QUEUE_DEPTH;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            g_state.gpu_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--interval-ms") == 0 && i + 1 < argc) {
            g_state.interval_ms = atof(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            g_state.output_path = argv[++i];
        } else if (strcmp(argv[i], "--queue-depth") == 0 && i + 1 < argc) {
            queue_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nvlinks") == 0 && i + 1 < argc) {
            g_state.nvlinks_to_poll = atoi(argv[++i]);
            if (g_state.nvlinks_to_poll < 1) g_state.nvlinks_to_poll = 1;
            if (g_state.nvlinks_to_poll > MAX_NVLINKS_PER_GPU) g_state.nvlinks_to_poll = MAX_NVLINKS_PER_GPU;
        } else if (strcmp(argv[i], "--quiet") == 0 || strcmp(argv[i], "-q") == 0) {
            g_state.quiet = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize queue
    if (queue_init(&g_state.queue, queue_depth) != 0) {
        fprintf(stderr, "Failed to initialize queue\n");
        return 1;
    }
    
    // Initialize NVML
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit failed: %s\n", nvmlErrorString(ret));
        queue_destroy(&g_state.queue);
        return 1;
    }
    
    // Get device handle
    nvmlDevice_t device;
    ret = nvmlDeviceGetHandleByIndex(g_state.gpu_index, &device);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(ret));
        nvmlShutdown();
        queue_destroy(&g_state.queue);
        return 1;
    }
    
    if (!g_state.quiet) {
        printf("Polling GPU %d, %d NVLinks, interval %.2f ms, output: %s\n",
               g_state.gpu_index, g_state.nvlinks_to_poll, g_state.interval_ms, g_state.output_path);
    }
    
    // Start writer thread
    pthread_t writer_thread;
    if (pthread_create(&writer_thread, NULL, writer_worker, NULL) != 0) {
        fprintf(stderr, "Failed to create writer thread\n");
        nvmlShutdown();
        queue_destroy(&g_state.queue);
        return 1;
    }
    
    // Run poll loop (blocks until signal)
    poll_loop(device);
    
    // Cleanup
    g_state.stop_flag = 1;
    pthread_join(writer_thread, NULL);
    nvmlShutdown();
    queue_destroy(&g_state.queue);
    
    if (!g_state.quiet) {
        printf("Done.\n");
    }
    return 0;
}

