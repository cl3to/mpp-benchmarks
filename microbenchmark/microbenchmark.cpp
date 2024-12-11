#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

int get_device(int dev_id) {
    int num_devices = omp_get_num_devices();
    return (4 + dev_id*4) % num_devices;
}

void check(uint64_t K, uint64_t N, int32_t numDevices, uint8_t *arr) {
    #pragma omp parallel for num_threads(numDevices)
    for (int32_t Device = 0; Device < numDevices; Device++) {
        #pragma omp target device(Device)
        {
            uint64_t checksum = 0;
            for (uint64_t i = 0; i < N; i++) {
                checksum += arr[i];
            }

            if (checksum == N) {
                printf("Device=%d: CORRECT SUM=%lu\n", Device, checksum);
            } else {
                printf("Device=%d: WRONG SUM=%lu\n", Device, checksum);
            }
        }
    }
}

double microbenchmark(uint64_t K, uint64_t N, int32_t numDevices, uint8_t *arr) {
    double start = 0, end = 0, total_time = 0;

    int rounds = (N <= 65536 ? 90 : 10);
    int warmup = 5;

    // Enter data asynchronously
    for (int32_t Device = 0; Device < numDevices; Device++) {
        for (uint64_t j = 0; j < K; j++) {
            #pragma omp target enter data map(alloc: arr[j*N:N]) device(Device) nowait
        }
    }

    // Ensure all target tasks have finished
    #pragma omp taskwait

    for(int i = 0; i < rounds+warmup; i++) {
        start = omp_get_wtime();

        // Enter data asynchronously
        for (int32_t Device = 0; Device < numDevices; Device++) {
            for (uint64_t j = 0; j < K; j++) {
                #pragma omp target update to(arr[j*N:N]) device(Device) nowait
            }
        }

        // Ensure all target tasks have finished
        #pragma omp taskwait

        end = omp_get_wtime();

        // does not consider the first attempts (warm-up)
        if (i >= warmup)
            total_time += (end - start);

        // printf("run#%d: %.4lf\n", i, (end - start));
    }

    // check if the data was sent correctly to the devices
    // check(K, N, numDevices, arr);

    // Enter data asynchronously
    for (int32_t Device = 0; Device < numDevices; Device++) {
        for (uint64_t j = 0; j < K; j++) {
            #pragma omp target exit data map(delete: arr[j*N:N]) device(Device) nowait
        }
    }
    // Ensure all target tasks have finished
    #pragma omp taskwait

    return total_time/rounds;
}

int main(int argc, char** argv) {

    if (argc < 4) {
        printf("You need pass the N, K, and D parameters.\nTry run ./microbenchmark N K D\n");
        return 1;
    }

    uint64_t N = atol(argv[1]);
    uint64_t K = atol(argv[2]);
    int32_t numDevices = atoi(argv[3]);
    // uint8_t *arr = (uint8_t*)malloc(K*N*sizeof(uint8_t));
    // uint8_t *arr = (uint8_t*)omp_target_alloc_host(K*N*sizeof(uint8_t), );
    uint8_t *arr = (uint8_t*)omp_alloc(K*N*sizeof(uint8_t), llvm_omp_target_host_mem_alloc);
    memset(arr, 1, K*N*sizeof(uint8_t));

    double avg_time = microbenchmark(K, N, numDevices, arr);
    double throughput = ((double)K*numDevices)/avg_time;

    printf("N: %lu\nK: %lu\nNumDevices: %d\n", N, K, numDevices);
    printf("AvgRuntime: %.6lfs\nAvgThroughput: %.4lf messages/s\n----\n", avg_time, throughput);
    // free(arr);
    omp_free(arr, llvm_omp_target_host_mem_alloc);
    return 0;
}
