#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <pthread.h>

#define OUTPUT_DIR "imgs_out"
#define MAX_PATH_LEN 1024
#define NTHREADS 4

struct Image { 
    char type_img[3];
    int width, height, max_color, datoxpixel;
    int* data; 
};

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
int IDX(const Image* im, int x, int y) { return (y * im->width + x) * im->datoxpixel; }

void ensure_output_dir() {
    struct stat st; if (stat(OUTPUT_DIR, &st) == -1) {
        if (mkdir(OUTPUT_DIR, 0755) == -1 && errno != EEXIST) { std::fprintf(stderr, "No se pudo crear '%s' (errno=%d)\n", OUTPUT_DIR, errno); }
    }
}

int g_counter = 1;
void build_unique(char* out, int cap, const char* filtro, const Image* im, const char* variante) {
    const char* ext = (im->datoxpixel == 3) ? "ppm" : "pgm"; int pid = (int)getpid(); long ts = (long)time(nullptr);
    std::snprintf(out, cap, "%s/%s_%s_%ld_%d_%d.%s", OUTPUT_DIR, filtro, variante, ts, pid, g_counter++, ext);
}



bool read_image(const char* path, Image* im) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "Error abriendo '%s'\n", path); return false; }
    if (std::fscanf(f, "%2s", im->type_img) != 1) { std::fclose(f); return false; }
    if (std::strcmp(im->type_img, "P2") != 0 && std::strcmp(im->type_img, "P3") != 0) { std::fprintf(stderr, "Formato no soportado\n"); std::fclose(f); return false; }
    if (std::fscanf(f, "%d %d", &im->width, &im->height) != 2) { std::fclose(f); return false; }
    if (std::fscanf(f, "%d", &im->max_color) != 1) { std::fclose(f); return false; }
    im->datoxpixel = (im->type_img[1] == '3') ? 3 : 1;

    long total = (long)im->width * im->height * im->datoxpixel;
    im->data = (int*)std::malloc(sizeof(int) * (size_t)total);
    if (!im->data) { std::fclose(f); return false; }
    for (long i = 0; i < total; ++i) {
        int v; if (std::fscanf(f, "%d", &v) != 1) { std::free(im->data); im->data = nullptr; std::fclose(f); return false; }
        im->data[i] = clampi(v, 0, im->max_color);
    }
    std::fclose(f); return true;
}

bool write_image(const char* path, const Image* im) {
    FILE* f = std::fopen(path, "w"); if (!f) { std::fprintf(stderr, "No se pudo escribir '%s'\n", path); return false; }
    std::fprintf(f, "%s\n%d %d\n%d\n", im->type_img, im->width, im->height, im->max_color);
    long total = (long)im->width * im->height * im->datoxpixel; int col = 0;
    for (long i = 0; i < total; ++i) { std::fprintf(f, "%d ", im->data[i]); if (++col >= 12) { std::fputc('\n', f); col = 0; } }
    if (col) std::fputc('\n', f); std::fclose(f); return true;
}

struct Task {
    const Image* in;
    Image* out;
    float K[3][3];
    bool normalize; // true en blur
    int y0, y1;     // [y0, y1)
};

void* worker(void* arg) {
    Task* t = (Task*)arg;
    const Image* in = t->in;
    Image* out = t->out;
    int W = in->width, H = in->height, C = in->datoxpixel;
    for (int y = t->y0; y < t->y1; ++y) {
        for (int x = 0; x < W; ++x) {
            if (C == 1) {
                float acc = 0.f, wsum = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            float w = t->K[ky + 1][kx + 1];
                            acc += in->data[IDX(in, nx, ny)] * w;
                            wsum += w;
                        }
                    }
                }
                float val = (t->normalize && wsum > 0.f) ? (acc / wsum) : acc;
                out->data[IDX(out, x, y)] = clampi((int)(val + 0.5f), 0, out->max_color);
            }
            else {
                float ar = 0.f, ag = 0.f, ab = 0.f, wsum = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            float w = t->K[ky + 1][kx + 1];
                            int p = IDX(in, nx, ny);
                            ar += in->data[p + 0] * w;
                            ag += in->data[p + 1] * w;
                            ab += in->data[p + 2] * w;
                            wsum += w;
                        }
                    }
                }
                if (t->normalize && wsum > 0.f) { ar /= wsum; ag /= wsum; ab /= wsum; }
                int o = IDX(out, x, y);
                out->data[o + 0] = clampi((int)(ar + 0.5f), 0, out->max_color);
                out->data[o + 1] = clampi((int)(ag + 0.5f), 0, out->max_color);
                out->data[o + 2] = clampi((int)(ab + 0.5f), 0, out->max_color);
            }
        }
    }
    return nullptr;
}

void kernel3x3_threads(const Image* in, Image* out, const float K[3][3], bool normalize){
    std::strncpy(out->type_img, in->type_img, 3);
    out->width = in->width;
    out->height = in->height;
    out->max_color = in->max_color;
    out->datoxpixel = in->datoxpixel;
    long total = (long)in->width * in->height * in->datoxpixel;
    out->data = (int*)std::malloc(sizeof(int) * (size_t)total);
    if (!out->data) {
        std::fprintf(stderr, "Sin memoria\n");
        return;
    }
    pthread_t th[NTHREADS];
    Task tasks[NTHREADS];

    int H = in->height;
    int chunk = (H + NTHREADS - 1) / NTHREADS;
    for (int i = 0; i < NTHREADS; ++i) {
        tasks[i].in = in; tasks[i].out = out; tasks[i].normalize = normalize;
        std::memcpy(tasks[i].K, K, sizeof(tasks[i].K));
        tasks[i].y0 = i * chunk; tasks[i].y1 = (tasks[i].y0 + chunk > H) ? H : tasks[i].y0 + chunk;
        pthread_create(&th[i], nullptr, worker, &tasks[i]);
    }
    for (int i = 0; i < NTHREADS; ++i) pthread_join(th[i], nullptr);
}

void usage(const char* p) {
    std::fprintf(stderr, "Uso:\n  %s <input.pgm|input.ppm> --f <blur|laplace|sharpen>\n", p);
}

int main(int argc, char* argv[]) {
    std::clock_t c0 = std::clock();     // tiempo de CPU
    std::time_t  w0 = std::time(NULL);  // wall-clock (segundos)
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* filtro = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--f") == 0 && i + 1 < argc) {
            filtro = argv[++i];
        }
    }
    if (!filtro) {
        usage(argv[0]);
        return 1;
    }

    const float K_blur[3][3] = { {1.f / 9,1.f / 9,1.f / 9},{1.f / 9,1.f / 9,1.f / 9},{1.f / 9,1.f / 9,1.f / 9} };
    const float K_laplace[3][3] = { {0,-1,0},{-1,4,-1},{0,-1,0} };
    const float K_sharpen[3][3] = { {0,-1,0},{-1,5,-1},{0,-1,0} };

    Image in = {}; if (!read_image(in_path, &in)) return 1;
    Image out = {};

    if (std::strcmp(filtro, "blur") == 0) {
        kernel3x3_threads(&in, &out, K_blur, true);
    }
    else if (std::strcmp(filtro, "laplace") == 0) {
        kernel3x3_threads(&in, &out, K_laplace, false);
    }
    else if (std::strcmp(filtro, "sharpen") == 0) {
        kernel3x3_threads(&in, &out, K_sharpen, false);
    }
    else {
        std::fprintf(stderr, "Filtro desconocido: %s\n", filtro);
        std::free(in.data);
        return 1;
    }
    std::clock_t c1 = std::clock();
    std::time_t  w1 = std::time(NULL);
    double cpu_ms = 1000.0 * (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
    double wall_s = std::difftime(w1, w0); // en segundos
    std::printf("Tiempo CPU: %.3f ms | Tiempo total: %.0f s\n", cpu_ms, wall_s);
    ensure_output_dir(); char outp[MAX_PATH_LEN]; build_unique(outp, MAX_PATH_LEN, filtro, &in, "pth");
    if (!write_image(outp, &out)) {
        std::fprintf(stderr, "No se pudo guardar\n");
        std::free(in.data);
        std::free(out.data);
        return 1;
    }
    std::printf("OK -> %s\n", outp);
    std::free(in.data); std::free(out.data);
    return 0;
}