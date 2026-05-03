#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/types.h>
#include <libgen.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <pci/pci.h>

#ifdef USE_NVIDIA
#include <nvml.h>
#endif

#ifdef USE_ROCM
#include <rocm_smi/rocm_smi.h>
#endif


#define MAX_CPUS 2
#define MAX_CPU_SENSORS 16
#define MAX_CPU_CORES 16
#define HWMON_BASE_PATH "/sys/class/hwmon"
#define AMD_CPU_DRIVER_NAME "k10temp"
#define INTEL_CPU_DRIVER_NAME "coretemp"
#define HWMON_SUFFIX_LABEL "label"
#define HWMON_SUFFIX_INPUT "input"

#define MAX_GPUS 4
#define PCI_BASE_PATH "/sys/bus/pci/devices"

typedef enum {
    VENDOR_UNKOWN = 0,
    VENDOR_NVIDIA = 0x10de,
    VENDOR_AMD    = 0x1002
} GPU_VENDOR;

typedef enum {
    INIT = 0,
    GET_HANDLE = 1,
    GET_NAME = 2,
    GET_TEMP = 3,
    GET_POWER_USAGE = 4,
    GET_POWER_MAX = 5,
    GET_MEMORY_INFO = 6,
    GET_UTIL_RATES = 7,
    GET_POWER_CONSTRAINTS = 8 ,
    GET_CUR_CLK_GRAPHICS = 9 ,
    GET_CUR_CLK_SM = 10,
    GET_CUR_CLK_MEM = 11,
    GET_MAX_CLK_GRAPHICS = 12 ,
    GET_MAX_CLK_SM = 13,
    GET_MAX_CLK_MEM = 14,
    GET_PSTATE = 15,
    GET_THROTTLE_REASONS = 16,
} nvml_err_src;

#define NV_VRAM_REGISTER_OFFSET 0x0000E2A8
#define NV_HOTSPOT_REGISTER_OFFSET 0x0002046C
#define PG_SZ sysconf(_SC_PAGE_SIZE)
#define MEM_PATH "/dev/mem"

#define MEM_INFO_PATH "/proc/meminfo"

typedef struct {
    char label[64];
    float value;
    int index;
    bool label_present;
    bool value_present;
} cpu_reading;

typedef struct {
    char driver[64];
    int cpu_id;
    char hwmon_path[512];
    cpu_reading readings[ MAX_CPU_SENSORS ];
} cpu_package;

typedef struct {
    cpu_package cpus[ MAX_CPUS ];
    int count;
} cpu_reading_list;

typedef struct {
    unsigned long long user, nice, system, idle;
    unsigned long long iowait, irq, softirq, steal;
} cpu_times;

typedef struct {
    cpu_times cores[ MAX_CPU_CORES ];
    cpu_times total;   // corresponds to "cpu " line
    int count;         // number of cores
} cpu_times_list;

typedef struct {
    char device_name[ 128 ];
    unsigned int temp_core;
    uint32_t temp_vram;
    uint32_t temp_junc;
    unsigned int pow_used;
    unsigned int pow_max;
    unsigned int pow_minlimit;
    unsigned int pow_maxlimit;
    unsigned long long mem_used;
    unsigned long long mem_total;
    unsigned long long mem_free;
    unsigned int cur_clk_graphics;
    unsigned int cur_clk_sm;
    unsigned int cur_clk_mem;
    unsigned int max_clk_graphics;
    unsigned int max_clk_sm;
    unsigned int max_clk_mem;
    unsigned int util_gpu;
    unsigned int util_memory;
    char pstate[ 2048 ];
    char thorttlereason[ 2048 ];
    char error_str[ 2048 ];
} nvml_readings;

typedef struct {
    char path[2048];
    float temp_core;
    float pow_used;
    int vram_used;
    int vram_total;
} hwmon_readings;

typedef struct {
    /* Identity */
    char pci_addr[32];   // "0000:01:00.0"
    uint16_t vendor_id;
    uint16_t device_id;

    /* Classification */
    char class_code[32];
    uint8_t base_class;
    uint8_t subclass;
    uint8_t prog_if;

    /* PCIe */
    char link_speed_current[32];
    char link_speed_max[32];

    int link_width_current;
    int link_width_max;

    /* abi readings */
    hwmon_readings hwmon_read;
    nvml_readings nvml_read;

    /* Metadata */
    char source[128];       // "nvidia", "amdgpu"
} gpu_reading;

typedef struct{
    gpu_reading readings[ MAX_GPUS ];
    int valid_cnt;
} gpu_reading_list;

typedef struct{
    size_t sys_total;
    size_t sys_free;
    size_t sys_used;
    size_t swap_total;
    size_t swap_free;
    size_t swap_used;
} ram_reading;

float read_sysfs_float(char *path) {
    FILE *float_file = fopen(path , "r");
    if (!float_file) return -1.0f;

    int float_reading;
    if ( fscanf(float_file , "%d" , &float_reading ) != 1 ) {
        fclose(float_file);
        return -1.0f;
    }

    fclose(float_file);
    return float_reading;
}

int read_sysfs_str(char *path , char *out , size_t size) {
    FILE *str_file = fopen(path , "r");
    if (!str_file) return -1;

    if (!fgets(out , size , str_file)) {
        fclose(str_file);
        return -1;
    }

    out[strcspn(out, "\n")] = 0;
    fclose(str_file);
    return 0;
}

int read_sysfs_int(char *path) {
    FILE *float_file = fopen(path , "r");
    if (!float_file) return -1;

    int int_reading;
    if ( fscanf(float_file , "%d" , &int_reading ) != 1 ) {
        fclose(float_file);
        return -1;
    }

    fclose(float_file);
    return int_reading;
}

static int read_register_temp(struct pci_dev *dev, uint32_t offset, uint32_t *temp) {
    int fd = open(MEM_PATH, O_RDWR | O_SYNC);
    if (fd < 0) {
        return -1;
    }

    uint32_t reg_addr = (dev->base_addr[0] & 0xFFFFFFFF) + offset;
    uint32_t base_offset = reg_addr & ~(PG_SZ-1);
    void *map_base = mmap(0, PG_SZ, PROT_READ, MAP_SHARED, fd, base_offset);
    if (map_base == MAP_FAILED) {
        close(fd);
        return -2;
    }

    uint32_t reg_value = *((uint32_t *)((char *)map_base + (reg_addr - base_offset)));

    if (offset == NV_HOTSPOT_REGISTER_OFFSET) {
        *temp = (reg_value >> 8) & 0xff;
    } else if (offset == NV_VRAM_REGISTER_OFFSET) {
        *temp = (reg_value & 0x00000fff) / 0x20;
    }

    munmap(map_base, PG_SZ);
    close(fd);

    return (*temp < 0x7f) ? 0 : -1;
}

bool mmio_read_u32(
    int fd,
    uint32_t phys_addr,
    size_t pg_size,
    uint32_t *out
) {
    uint32_t base = phys_addr & ~(pg_size - 1);
    uint32_t offset = phys_addr - base;

    void *map_base = mmap(NULL, pg_size, PROT_READ, MAP_SHARED, fd, base);
    if (map_base == MAP_FAILED) {
        return false;
    }

    *out = *((uint32_t *)((char *)map_base + offset));

    munmap(map_base, pg_size);
    return true;
}

cpu_reading_list read_cpu_hwmon() {
    cpu_reading_list list = {0};

    DIR *base_dir = opendir( HWMON_BASE_PATH );
    if (!base_dir) return list;

    struct dirent *base_entry;

    while ((base_entry = readdir(base_dir)) != NULL) {
        if ( strncmp( base_entry->d_name , "hwmon" , 5 ) == 0  ) {

            char hwmon_path[512];
            char hwmon_path_name[1024];

           snprintf(
               hwmon_path,
               sizeof(hwmon_path),
               "%s/%s" ,
               HWMON_BASE_PATH , base_entry->d_name );

            DIR *hwmon_dir = opendir( hwmon_path );
            if (!hwmon_dir) continue;

            snprintf(
                hwmon_path_name,
                sizeof(hwmon_path_name),
                "%s/name", hwmon_path );

            char driver[64];
            int read_res = read_sysfs_str( hwmon_path_name, driver, sizeof(driver) );
            int is_amdcpu = strcmp(driver , AMD_CPU_DRIVER_NAME) == 0;
            int is_intelcpu = strcmp(driver , INTEL_CPU_DRIVER_NAME) == 0;

            if ( read_res != 0 || ( !is_amdcpu && !is_intelcpu ) ) {
                closedir(hwmon_dir);
                continue;
            }

            cpu_package *cpu = &list.cpus[ list.count ];
            snprintf( cpu->driver , sizeof(cpu->driver) , "%s" , driver );
            snprintf( cpu->hwmon_path , sizeof(cpu->hwmon_path) , "%s" , hwmon_path);
            cpu->cpu_id = list.count;
            list.count++;

            if ( list.count >= MAX_CPUS ) {
                closedir(hwmon_dir);
                continue;
            }

            struct dirent *hwmon_dir_entry;

            while ( (hwmon_dir_entry = readdir(hwmon_dir)) != NULL ) {
                if ( strncmp( hwmon_dir_entry->d_name , "temp" , 4) == 0 ) {

                    int index;
                    char type[ 16 ];

                    if ( sscanf( hwmon_dir_entry->d_name , "temp%d_%15s" , &index , type ) == 2 ) {

                        if (index >= MAX_CPU_SENSORS) continue;

                        cpu_reading *reading = &cpu->readings[index];
                        char hwmon_full_path[ 1024 ];
                        reading->label_present = false;
                        reading->value_present = false;

                        snprintf(
                            hwmon_full_path,
                            sizeof(hwmon_full_path),
                            "%s/%s" , hwmon_path , hwmon_dir_entry->d_name);

                        reading->index = index;

                        if (strcmp(type, HWMON_SUFFIX_INPUT) == 0 ) {
                            reading->value = read_sysfs_float(hwmon_full_path) / 1000.0f;
                            reading->label_present = true;
                        } else if ( strcmp(type , HWMON_SUFFIX_LABEL) == 0  ) {
                            read_sysfs_str(
                                hwmon_full_path ,
                                reading->label ,
                                sizeof(reading->label)
                            );
                            reading->value_present = true;
                        }
                    }
                }
            }
            closedir(hwmon_dir);
        }
    }

    closedir(base_dir);
    return list;
}

float compute_usage(cpu_times *a, cpu_times *b) {
    unsigned long long idle_a = a->idle + a->iowait;
    unsigned long long idle_b = b->idle + b->iowait;

    unsigned long long total_a =
        a->user + a->nice + a->system + a->idle +
        a->iowait + a->irq + a->softirq + a->steal;

    unsigned long long total_b =
        b->user + b->nice + b->system + b->idle +
        b->iowait + b->irq + b->softirq + b->steal;

    unsigned long long delta_total = total_b - total_a;
    unsigned long long delta_idle  = idle_b - idle_a;

    if (delta_total == 0) return 0.0f;

    return (float)(delta_total - delta_idle) / delta_total * 100.0f;
}

int read_cpu_times(cpu_times_list *list) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return -1;

    char line[256];
    list->count = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu", 3) != 0)
            break;

        cpu_times t = {0};

        sscanf(line, "%*s %llu %llu %llu %llu %llu %llu %llu %llu",
               &t.user, &t.nice, &t.system, &t.idle,
               &t.iowait, &t.irq, &t.softirq, &t.steal);

        if (line[3] == ' ') {
            // aggregate line
            list->total = t;
        } else {
            if (list->count < MAX_CPU_CORES) {
                list->cores[list->count++] = t;
            }
        }
    }

    fclose(f);
    return 0;
}

ram_reading read_ram_usage() {
    FILE *f = fopen( MEM_INFO_PATH, "r");

    ram_reading reading = { 0 };

    if (!f) { return reading; }

    char line[512];
    long value;

    while (fgets(line, sizeof(line) , f)) {
        if ( sscanf(line, "MemTotal: %ld kB", &value) == 1 ) {
            reading.sys_total = value;
        } else if ( sscanf(line , "MemAvailable: %ld kB" , &value) == 1 ) {
            reading.sys_free = value;
        } else if ( sscanf(line, "SwapTotal: %ld kB", &value) == 1 ) {
            reading.swap_total = value;
        } else if ( sscanf(line, "SwapFree: %ld kB", &value) == 1 ) {
            reading.swap_free = value;
        }
    }

    fclose( f );

    reading.sys_used = reading.sys_total - reading.sys_free;
    reading.swap_used = reading.swap_total - reading.swap_free;

    return reading;
}

bool hwmon_chk(char * path , char *out , size_t size) {
    DIR *base_dir = opendir( path );
    if (!base_dir) return false;

    struct dirent *base_entry;

    while ((base_entry = readdir( base_dir )) != NULL) {
        if ( strncmp( base_entry->d_name , "hwmon" , 5 ) == 0 ) {
            snprintf( out , size , "%s/%s" , path , base_entry->d_name);
            closedir(base_dir);
            return true;
        }
    }

    closedir(base_dir);
    return true;
}

hwmon_readings read_gpu_hwmon( char *path , uint16_t vendor_id ) {

    hwmon_readings ret_read = {0};

    if (vendor_id == VENDOR_AMD) {

        char pci_path_hwmon_temp[ 2048 ];
        char pci_path_hwmon_watts[ 2048 ];

        snprintf(
            pci_path_hwmon_temp ,
            sizeof(pci_path_hwmon_temp) ,
            "%s/temp1_input" , path );

        snprintf(
            pci_path_hwmon_watts ,
            sizeof(pci_path_hwmon_watts) ,
            "%s/power1_input", path);

        ret_read.temp_core = read_sysfs_float( pci_path_hwmon_temp );
        ret_read.pow_used = read_sysfs_float( pci_path_hwmon_watts ) / 1000.0f;
        snprintf( ret_read.path , sizeof(ret_read.path), "%s" , path );
    }

    return ret_read;
}

void nvml_error_handler(
    nvmlReturn_t ret , char *err_buff , size_t size , nvml_err_src err ) {

        switch ( err ) {
            case INIT :
                snprintf( err_buff , size , "Init : %s" , nvmlErrorString( ret ) );
                break;
            case GET_HANDLE :
                snprintf( err_buff , size , "Get Handle : %s" , nvmlErrorString( ret ) );
                break;
            case GET_NAME :
                snprintf( err_buff , size , "Get Name : %s" , nvmlErrorString( ret ) );
                break;
            case GET_POWER_USAGE :
                snprintf( err_buff , size , "Get Current Power : %s" , nvmlErrorString( ret ) );
                break;
            case GET_POWER_MAX :
                snprintf( err_buff , size , "Get Max Power : %s" , nvmlErrorString( ret ) );
                break;
            case GET_TEMP :
                snprintf( err_buff , size , "Get Temperature : %s" , nvmlErrorString( ret ) );
                break;
            case GET_MEMORY_INFO :
                snprintf( err_buff , size , "Get Memory : %s" , nvmlErrorString( ret ) );
                break;
            case GET_UTIL_RATES :
                snprintf( err_buff , size , "Get Utilization : %s" , nvmlErrorString( ret ) );
                break;
            case GET_POWER_CONSTRAINTS :
                snprintf( err_buff , size , "Get Power Constraints : %s" , nvmlErrorString( ret ) );
                break;
            case GET_CUR_CLK_GRAPHICS :
                snprintf( err_buff , size , "Get Current Graphics Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_CUR_CLK_MEM :
                snprintf( err_buff , size , "Get Current Memory Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_CUR_CLK_SM :
                snprintf( err_buff , size , "Get Current SM Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_MAX_CLK_GRAPHICS :
                snprintf( err_buff , size , "Get Max Graphics Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_MAX_CLK_MEM :
                snprintf( err_buff , size , "Get Max Memory Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_MAX_CLK_SM :
                snprintf( err_buff , size , "Get Max SM Clock : %s" , nvmlErrorString( ret ) );
                break;
            case GET_PSTATE :
                snprintf( err_buff , size , "Get P-State : %s" , nvmlErrorString( ret ) );
                break;
            case GET_THROTTLE_REASONS :
                snprintf( err_buff , size , "Get Throttle Reason : %s" , nvmlErrorString( ret ) );
                break;
            default:
                snprintf( err_buff , size , "%s" , nvmlErrorString( ret ) );
        }
}

void process_nvml_pstate(nvmlPstates_t p_state , char *out , size_t size) {

    if (!out || size == 0) return;

       switch (p_state) {
           case NVML_PSTATE_0:
               snprintf(out, size, "P0 Max Performance");
               break;

           case NVML_PSTATE_1:
           case NVML_PSTATE_2:
           case NVML_PSTATE_3:
           case NVML_PSTATE_4:
           case NVML_PSTATE_5:
           case NVML_PSTATE_6:
           case NVML_PSTATE_7:
               snprintf(out, size, "P1-P7 Boost / Active");
               break;

           case NVML_PSTATE_8:
           case NVML_PSTATE_9:
           case NVML_PSTATE_10:
           case NVML_PSTATE_11:
           case NVML_PSTATE_12:
           case NVML_PSTATE_13:
           case NVML_PSTATE_14:
           case NVML_PSTATE_15:
               snprintf(out, size, "P8-P15 Idle / Power Saving");
               break;

           default:
               snprintf(out, size, "Unknown");
               break;
       }
}

void process_nvml_throttlereason(
    unsigned long long throttlereason , char * out , size_t size ) {

        switch ( throttlereason ) {
            case 0x0000000000000001LL:
                snprintf(out , size , "Idling");
                break;
            case 0x0000000000000002LL:
                snprintf(out , size , "Application Defined");
                break;
            case 0x0000000000000004LL:
                snprintf(out , size , "SW Power Cap");
                break;
            case 0x0000000000000008LL:
                snprintf(out , size , "HW Slow down");
                break;
            case 0x0000000000000010LL:
                snprintf(out , size , "Sync Boost");
                break;
            case 0x0000000000000020LL:
                snprintf(out , size , "SW Thermal Slowdown");
                break;
            case 0x0000000000000040LL:
                snprintf(out , size , "HW Thermal Slowdown");
                break;
            case 0x0000000000000080LL:
                snprintf(out , size , "HW Power Break");
                break;
            case 0x0000000000000100LL:
                snprintf(out , size , "Display Clock");
                break;
        }

}

nvml_readings read_gpu_nvml( char *bus_id ) {
    nvml_readings ret_read = {0};
    nvmlReturn_t ret;

    ret = nvmlInit_v2();
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 0
        );
        nvmlShutdown();
        return ret_read;
    }

    nvmlDevice_t device;

    ret = nvmlDeviceGetHandleByPciBusId_v2( bus_id, &device);
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 1
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetName(
        device , ret_read.device_name , sizeof( ret_read.device_name )
    );
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 2
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetTemperature( device , 0 , &ret_read.temp_core);
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 3
        );
        nvmlShutdown();
        return ret_read;
    }

    // nvml returns mW
    ret = nvmlDeviceGetPowerUsage( device , &ret_read.pow_used);
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 4
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetEnforcedPowerLimit( device , &ret_read.pow_max );
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 5
        );
        nvmlShutdown();
        return ret_read;
    }

    nvmlMemory_v2_t mem = {0};
    mem.version = nvmlMemory_v2;
    ret = nvmlDeviceGetMemoryInfo_v2( device , &mem);

    // nvml returns mebibytes
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 6
        );
        nvmlShutdown();
        return ret_read;
    }

    ret_read.mem_used = mem.used;
    ret_read.mem_total = mem.total;
    ret_read.mem_free = mem.free;

    nvmlUtilization_t util = {0};
    ret = nvmlDeviceGetUtilizationRates( device , &util);

    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 7
        );
        nvmlShutdown();
        return ret_read;
    }

    ret_read.util_gpu = util.gpu;
    ret_read.util_memory = util.memory;

    ret = nvmlDeviceGetPowerManagementLimitConstraints(
        device , &ret_read.pow_minlimit , &ret_read.pow_maxlimit );
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 8
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetClockInfo( device , 0 , &ret_read.cur_clk_graphics);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 9
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetClockInfo( device , 1 , &ret_read.cur_clk_sm);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 10
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetClockInfo( device , 2 , &ret_read.cur_clk_mem);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 11
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetMaxCustomerBoostClock( device , 0 , &ret_read.max_clk_graphics);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 12
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetMaxCustomerBoostClock( device , 1 , &ret_read.max_clk_sm);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 13
        );
        nvmlShutdown();
        return ret_read;
    }

    ret = nvmlDeviceGetMaxCustomerBoostClock( device , 2 , &ret_read.max_clk_mem);
    if ( ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 14
        );
        nvmlShutdown();
        return ret_read;
    }

    nvmlPstates_t p_state;
    ret = nvmlDeviceGetPerformanceState( device , &p_state);
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 15
        );
        nvmlShutdown();
        return ret_read;
    }

    process_nvml_pstate( p_state , ret_read.pstate , sizeof( ret_read.pstate ));

    unsigned long long clk_ev_reasons;
    ret = nvmlDeviceGetCurrentClocksThrottleReasons( device , &clk_ev_reasons);
    if ( ret != NVML_SUCCESS ) {
        nvml_error_handler(
            ret , ret_read.error_str , sizeof( ret_read.error_str ) , 16
        );
        nvmlShutdown();
        return ret_read;
    }

    process_nvml_throttlereason(
        clk_ev_reasons , ret_read.thorttlereason, sizeof( ret_read.thorttlereason )
    );

    nvmlShutdown();
    return ret_read;
}

gpu_reading_list gpu_telemetry() {
    struct pci_access *pacc;
    struct pci_dev *dev;
    gpu_reading_list list = {0};
    int device_cnt = 0;

    pacc = pci_alloc();
    pci_init(pacc);
    pci_scan_bus(pacc);

    for (dev = pacc->devices; dev; dev = dev->next) {

        pci_fill_info(dev,
            PCI_FILL_IDENT |
            PCI_FILL_CLASS |
            PCI_FILL_SUBSYS |
            PCI_FILL_DRIVER |
            PCI_FILL_BASES
        );

        /* Class (IMPORTANT for GPU detection) */
        unsigned int base_class = (dev->device_class >> 8) & 0xff;
        unsigned int subclass   = dev->device_class & 0xff;
        unsigned int prog_if    = dev->prog_if;

        if ( base_class != 0x03 )
            continue;

        gpu_reading * gpu = &list.readings[ device_cnt ];
        char path[256];

        if (device_cnt >= MAX_GPUS)
            break;

        snprintf(
            gpu->pci_addr ,
            sizeof(gpu->pci_addr),
            "%04x:%02x:%02x.%d", dev->domain, dev->bus, dev->dev, dev->func
        );
        gpu->vendor_id = dev->vendor_id;
        gpu->device_id = dev->device_id;
        snprintf(
            gpu->class_code,
            sizeof(gpu->class_code),
            "%02x:%02x:%02x" , base_class, subclass, prog_if);
        gpu->base_class = base_class;
        gpu->subclass = subclass;
        gpu->prog_if = prog_if;

        char pci_path[512];

        snprintf(
            pci_path ,
            sizeof(pci_path) ,
            "%s/%s", PCI_BASE_PATH , gpu->pci_addr );

        char pci_path_currlnkspd[1024];

        snprintf(
            pci_path_currlnkspd ,
            sizeof(pci_path_currlnkspd) ,
            "%s/current_link_speed" , pci_path);

        read_sysfs_str( pci_path_currlnkspd, gpu->link_speed_current , sizeof( gpu->link_speed_current ));

        char pci_path_maxlnkspd[1024];

        snprintf(
            pci_path_maxlnkspd ,
            sizeof(pci_path_maxlnkspd) ,
            "%s/max_link_speed" , pci_path);

        read_sysfs_str( pci_path_maxlnkspd, gpu->link_speed_max , sizeof( gpu->link_speed_max ));

        char pci_path_currlnkwdt[1024];

        snprintf(
            pci_path_currlnkwdt ,
            sizeof(pci_path_currlnkwdt) ,
            "%s/current_link_width" , pci_path);

        gpu->link_width_current = read_sysfs_int( pci_path_currlnkwdt );

        char pci_path_maxlnkwdt[1024];

        snprintf(
            pci_path_maxlnkwdt ,
            sizeof(pci_path_maxlnkwdt) ,
            "%s/max_link_width" , pci_path);

        gpu->link_width_max = read_sysfs_int( pci_path_maxlnkwdt );

        char pci_path_hwmon[ 1024 ];
        char hwmon_path[ 2048 ];

        snprintf(
            pci_path_hwmon ,
            sizeof(pci_path_hwmon) ,
            "%s/hwmon" , pci_path );

        if ( hwmon_chk( pci_path_hwmon , hwmon_path , sizeof(hwmon_path) ) ) {
            gpu->hwmon_read = read_gpu_hwmon( hwmon_path , gpu->vendor_id);

            if ( gpu->vendor_id == VENDOR_AMD ) {
                char pci_path_vramused[1024];

                snprintf(
                    pci_path_vramused ,
                    sizeof(pci_path_vramused) ,
                    "%s/mem_info_vram_used" , pci_path);

                gpu->hwmon_read.vram_used = read_sysfs_int( pci_path_vramused );

                char pci_path_vramtotal[1024];

                snprintf(
                    pci_path_vramtotal ,
                    sizeof(pci_path_vramtotal) ,
                    "%s/mem_info_vram_total" , pci_path);

                gpu->hwmon_read.vram_total = read_sysfs_int( pci_path_vramtotal );
            }

            snprintf( gpu->source , sizeof( gpu->source ) , "hwmon");
            list.valid_cnt++;
        } else {

            if ( gpu->vendor_id == VENDOR_NVIDIA ) {
                gpu->nvml_read = read_gpu_nvml( gpu->pci_addr );

                bool is_root = ( getuid() == 0 );
                list.valid_cnt++;

                if ( !is_root ) {
                    snprintf( gpu->source , sizeof( gpu->source ) , "nvml");
                    gpu->nvml_read.temp_vram = 0;
                    gpu->nvml_read.temp_junc = 0;
                } else {
                    snprintf( gpu->source , sizeof( gpu->source ) , "nvml+MMIO");
                    int fd = open(MEM_PATH, O_RDWR | O_SYNC);
                    if (fd < 0) {
                        gpu->nvml_read.temp_vram = -1;
                        gpu->nvml_read.temp_junc = -1;
                        continue;
                    }

                    uint32_t phys_vram = (dev->base_addr[0] & 0xFFFFFFFF) + NV_VRAM_REGISTER_OFFSET;
                    uint32_t phys_junc = (dev->base_addr[0] & 0xFFFFFFFF) + NV_HOTSPOT_REGISTER_OFFSET;
                    uint32_t vram_raw;

                    if (mmio_read_u32(fd, phys_vram, PG_SZ, &vram_raw)) {
                        gpu->nvml_read.temp_vram = (float)(vram_raw & 0x00000fff) / 0x20;
                    } else {
                        gpu->nvml_read.temp_vram = -2;
                    }

                    uint32_t hotspot_raw;

                    if (mmio_read_u32(fd, phys_junc, PG_SZ, &hotspot_raw)) {
                        uint32_t temp = (hotspot_raw >> 8) & 0xff;
                        gpu->nvml_read.temp_junc = (temp < 0x7f) ? (float)temp : -3;
                    } else {
                        gpu->nvml_read.temp_junc =-2;
                    }
                }
            }
        }
        device_cnt += 1;
    }

    pci_cleanup(pacc);
    return list;
}

void print_jspn_file(
    FILE *fptr ,
    cpu_reading_list *cpulist ,
    cpu_times_list *prev , cpu_times_list *curr ,
    gpu_reading_list *gpulist ,
    ram_reading *ramread) {
        fprintf(fptr , "{\n");
        fprintf(fptr , "\t\"cpus\": [\n");

        for (int i = 0; i < cpulist->count; i++) {
            cpu_package *cpu = &cpulist->cpus[i];

            fprintf(fptr , "\t\t{\n");
            fprintf(fptr , "\t\t\t\"driver\": \"%s\",\n", cpu->driver);
            fprintf(fptr , "\t\t\t\"cpu_id\": %d,\n", cpu->cpu_id);
            fprintf(fptr , "\t\t\t\"hwmon_path\": \"%s\",\n", cpu->hwmon_path);
            fprintf(fptr , "\t\t\t\"sensors\": [\n");

            int first_sensor = 1;

            for (int j = 0; j < MAX_CPU_SENSORS; j++) {
                cpu_reading *r = &cpu->readings[j];

                if (!r->value_present && !r->label_present)
                    continue;

                if (!first_sensor) {
                    fprintf(fptr , ",\n");
                }

                fprintf(fptr , "\t\t\t\t{\n");
                fprintf(fptr , "\t\t\t\t\t\"index\": %d,\n", r->index);
                fprintf(fptr , "\t\t\t\t\t\"label\": \"%s\",\n", r->label);
                fprintf(fptr , "\t\t\t\t\t\"value\": %.3f\n", r->value);
                fprintf(fptr , "\t\t\t\t}");

                first_sensor = 0;
            }

            fprintf(fptr , "\n\t\t\t]\n");

            if (i < cpulist->count - 1)
                fprintf(fptr , "\t\t},\n");
            else
                fprintf(fptr , "\t\t}\n");
        }

        fprintf(fptr , "\t],\n");
        fprintf(fptr , "\t\"usage\" : {\n");
        fprintf(fptr , "\t\t\"cpu\" : {\n");
        fprintf(fptr , "\t\t\t\"total\" : %.2f ,\n" , compute_usage(&prev->total, &curr->total) );
        fprintf(fptr , "\t\t\t\"cores\" : [ \n");

        for (int i = 0; i < curr->count; i++) {
            fprintf(fptr , "\t\t\t\t%.2f" , compute_usage(&prev->cores[i], &curr->cores[i]) );
            if ( i + 1 != curr->count) {
                fprintf(fptr , ",\n");
            } else {
                fprintf(fptr ,  "\n" );
            }
        }

        fprintf(fptr , "\t\t\t]\n");
        fprintf(fptr , "\t\t},\n");
        fprintf(fptr , "\t\t\"ram\" : {\n");

        fprintf(fptr ,  "\t\t\t\"used\" : %zu ,\n" , ramread->sys_used / 1000 );
        fprintf(fptr ,  "\t\t\t\"total\" : %zu ,\n" , ramread->sys_total / 1000 );
        fprintf(fptr ,  "\t\t\t\"free\" : %zu ,\n" , ramread->sys_free / 1000 );
        fprintf(fptr ,  "\t\t\t\"swap_used\" : %zu ,\n" , ramread->swap_used / 1000 );
        fprintf(fptr ,  "\t\t\t\"swap_total\" : %zu ,\n" , ramread->swap_total / 1000 );
        fprintf(fptr ,  "\t\t\t\"swap_free\" : %zu\n" , ramread->swap_free / 1000 );


        fprintf(fptr , "\t\t}\n");
        fprintf(fptr , "\t},\n");
        fprintf(fptr , "\t\"gpu_info\" : [\n");

        for (int i=0 ; i < MAX_GPUS ; i++) {

            if (gpulist->readings[i].pci_addr[0] == '\0') {
                continue;
            }

            fprintf(fptr , "\t\t{\n");
            fprintf(fptr , "\t\t\t\"pci_addr\" : \"%s\" ,\n" , gpulist->readings[i].pci_addr );
            fprintf(fptr , "\t\t\t\"class_code\" : \"%s\" ,\n" , gpulist->readings[i].class_code );
            fprintf(fptr , "\t\t\t\"curr_link_speed\" : \"%s\",\n" , gpulist->readings[i].link_speed_current );
            fprintf(fptr , "\t\t\t\"max_link_speed\" : \"%s\",\n" , gpulist->readings[i].link_speed_max );
            fprintf(fptr , "\t\t\t\"curr_link_width\" : %d,\n" , gpulist->readings[i].link_width_current );
            fprintf(fptr , "\t\t\t\"max_link_width\" : %d,\n" , gpulist->readings[i].link_width_max );

            if ( strncmp( gpulist->readings[i].source , "hwmon" , 3 ) == 0 ) {
                fprintf(fptr , "\t\t\t\"hwmon_path\" : \"%s\" ,\n" , gpulist->readings[i].hwmon_read.path );
                fprintf(fptr , "\t\t\t\"core_temp\" : %.0f ,\n" , gpulist->readings[i].hwmon_read.temp_core / 1000.0f );
                fprintf(fptr , "\t\t\t\"power_used\" : %.0f ,\n" , gpulist->readings[i].hwmon_read.pow_used / 1000.0f );
                fprintf(fptr , "\t\t\t\"vram_used\" : %d ,\n" , gpulist->readings[i].hwmon_read.vram_used / (1024 * 1024) );
                fprintf(fptr , "\t\t\t\"vram_total\" : %d\n" , gpulist->readings[i].hwmon_read.vram_total / (1024 * 1024) );
            } else if ( strncmp( gpulist->readings[i].source , "nvml" , 3) == 0 ) {
                fprintf(fptr , "\t\t\t\"gpu_name\" : \"%s\",\n" , gpulist->readings[i].nvml_read.device_name );
                fprintf(fptr , "\t\t\t\"core_temp\" : %d,\n" , gpulist->readings[i].nvml_read.temp_core );
                fprintf(fptr , "\t\t\t\"vram_temp\" : %u,\n" , gpulist->readings[i].nvml_read.temp_vram );
                fprintf(fptr , "\t\t\t\"junc_temp\" : %u,\n" , gpulist->readings[i].nvml_read.temp_junc );
                fprintf(fptr , "\t\t\t\"power_used\" : %d,\n" , gpulist->readings[i].nvml_read.pow_used / 1000 );
                fprintf(fptr , "\t\t\t\"power_max\" : %d,\n" , gpulist->readings[i].nvml_read.pow_max / 1000 );
                fprintf(fptr , "\t\t\t\"power_limit_min\" : %d,\n" , gpulist->readings[i].nvml_read.pow_minlimit / 1000 );
                fprintf(fptr , "\t\t\t\"power_limit_max\" : %d,\n" , gpulist->readings[i].nvml_read.pow_maxlimit / 1000 );
                fprintf(fptr , "\t\t\t\"vram_used\" : %llu,\n" , gpulist->readings[i].nvml_read.mem_used / ( 1024 * 1024 ) );
                fprintf(fptr , "\t\t\t\"vram_total\" : %llu,\n" , gpulist->readings[i].nvml_read.mem_total / ( 1024 * 1024 ) );
                fprintf(fptr , "\t\t\t\"vram_free\" : %llu,\n" , gpulist->readings[i].nvml_read.mem_free / ( 1024 * 1024 ) );
                fprintf(fptr , "\t\t\t\"curr_clk_graphics\" : %d,\n" , gpulist->readings[i].nvml_read.cur_clk_graphics );
                fprintf(fptr , "\t\t\t\"curr_clk_sm\" : %d,\n" , gpulist->readings[i].nvml_read.cur_clk_sm );
                fprintf(fptr , "\t\t\t\"curr_clk_mem\" : %d,\n" , gpulist->readings[i].nvml_read.cur_clk_mem );
                fprintf(fptr , "\t\t\t\"max_clk_graphics\" : %d,\n" , gpulist->readings[i].nvml_read.max_clk_graphics );
                fprintf(fptr , "\t\t\t\"max_clk_sm\" : %d,\n" , gpulist->readings[i].nvml_read.max_clk_sm );
                fprintf(fptr , "\t\t\t\"max_clk_mem\" : %d,\n" , gpulist->readings[i].nvml_read.max_clk_mem );
                fprintf(fptr , "\t\t\t\"util_gpu\" : %d,\n" , gpulist->readings[i].nvml_read.util_gpu );
                fprintf(fptr , "\t\t\t\"util_mem\" : %d,\n" , gpulist->readings[i].nvml_read.util_memory );
                fprintf(fptr , "\t\t\t\"power_state\" : \"%s\",\n" , gpulist->readings[i].nvml_read.pstate );
                fprintf(fptr , "\t\t\t\"throttle_reason\" : \"%s\",\n" , gpulist->readings[i].nvml_read.thorttlereason );
                fprintf(fptr , "\t\t\t\"error_str\" : \"%s\"\n" , gpulist->readings[i].nvml_read.error_str );
            }

            if ( i + 1 != gpulist->valid_cnt ) {
                fprintf(fptr ,  "\t\t},\n" );
            } else {
                fprintf(fptr ,  "\t\t}\n" );
            }
        }

        fprintf(fptr , "\t]\n");
        fprintf(fptr , "}\n");
}

void print_call() {
    cpu_times_list prev , curr;
    read_cpu_times( &prev );
    sleep( 1 );
    read_cpu_times( &curr );
    cpu_reading_list cpulist = read_cpu_hwmon();
    gpu_reading_list gpulist = gpu_telemetry();
    ram_reading ramread = read_ram_usage();
    FILE *f = fopen("/var/lib/telemetryd/data.json", "w");
    print_jspn_file(f , &cpulist, &prev, &curr , &gpulist , &ramread);
    fclose( f );
}

int main() {
    print_call();
    return 0;
}
