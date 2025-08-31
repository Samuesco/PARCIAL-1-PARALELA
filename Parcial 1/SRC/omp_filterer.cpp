#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <omp.h>

#define OUTPUT_DIR "imgs_out"
#define MAX_PATH_LEN 1024

struct Image {
    char type_img[3];   // "P2" o "P3"
    int  width, height;
    int  max_color;     // p.ej., 255
    int  unidadxpixel;  // 1 para P2, 3 para P3
    int* data;          // width*height*unidadxpixel
};

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
int IDX(const Image* im, int x, int y) { return (y * im->width + x) * im->unidadxpixel; }

// base del nombre a partir de la ruta (sin carpetas ni extensión)
void base_from_path(const char* in, char* out, int cap) {
    const char* p = in + std::strlen(in);
    while (p > in && p[-1] != '/' && p[-1] != '\\') --p;   // salta carpetas
    std::snprintf(out, cap, "%s", p);                      // copia nombre
    int n = (int)std::strlen(out);
    for (int i = n - 1; i >= 0; --i) {                      // corta extensión
        if (out[i] == '.') { out[i] = 0; break; }
        if (out[i] == '/' || out[i] == '\\') break;
    }
}

// crea imgs_out/<base>_<filtro>_<ts>.(pgm|ppm)
void build_named(char* out, int cap, const char* base, const char* filtro, const Image* im) {
    const char* ext = (im->unidadxpixel == 3) ? "ppm" : "pgm";
    long ts = (long)time(NULL);
    std::snprintf(out, cap, "%s/%s_%s_%ld.%s", OUTPUT_DIR, base, filtro, ts, ext);
}

void ensure_output_dir() {
    struct stat st;
    if (stat(OUTPUT_DIR, &st) == -1) {
        if (mkdir(OUTPUT_DIR, 0755) == -1 && errno != EEXIST) {
            std::fprintf(stderr, "No se pudo crear '%s' (errno=%d)\n", OUTPUT_DIR, errno);
        }
    }
}

// Lectura P2/P3 ASCII (valores separados por espacio; sin comentarios).
bool read_image(const char* path, Image* im) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "Error abriendo '%s'\n", path); return false; }

    if (std::fscanf(f, "%2s", im->type_img) != 1) { std::fclose(f); return false; }
    if (std::strcmp(im->type_img, "P2") != 0 && std::strcmp(im->type_img, "P3") != 0) {
        std::fprintf(stderr, "Formato no soportado (P2/P3 ASCII)\n"); std::fclose(f); return false;
    }
    if (std::fscanf(f, "%d %d", &im->width, &im->height) != 2) { std::fclose(f); return false; }
    if (std::fscanf(f, "%d", &im->max_color) != 1) { std::fclose(f); return false; }

    im->unidadxpixel = (im->type_img[1] == '3') ? 3 : 1;
    long total = (long)im->width * im->height * im->unidadxpixel;
    im->data = (int*)std::malloc(sizeof(int) * (size_t)total);
    if (!im->data) { std::fclose(f); return false; }

    for (long i = 0; i < total; ++i) {
        int v; if (std::fscanf(f, "%d", &v) != 1) { std::free(im->data); im->data = NULL; std::fclose(f); return false; }
        im->data[i] = clampi(v, 0, im->max_color);
    }
    std::fclose(f);
    return true;
}

bool write_image(const char* path, const Image* im) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "No se pudo escribir '%s'\n", path); return false; }
    std::fprintf(f, "%s\n%d %d\n%d\n", im->type_img, im->width, im->height, im->max_color);
    long total = (long)im->width * im->height * im->unidadxpixel;
    int col = 0;
    for (long i = 0; i < total; ++i) {
        std::fprintf(f, "%d ", im->data[i]);
        if (++col >= 12) { std::fputc('\n', f); col = 0; }
    }
    if (col) std::fputc('\n', f);
    std::fclose(f);
    return true;
}

// Convolución 3×3 estilo tu snippet (blur normaliza con weight_sum; P3 procesa R,G,B juntos).
void apply_kernel3x3(const Image* in, Image* out, const float K[3][3], bool normalize) {
    std::strncpy(out->type_img, in->type_img, 3);
    out->width = in->width; out->height = in->height;
    out->max_color = in->max_color; out->unidadxpixel = in->unidadxpixel;

    long total = (long)in->width * in->height * in->unidadxpixel;
    out->data = (int*)std::malloc(sizeof(int) * (size_t)total);
    if (!out->data) { std::fprintf(stderr, "Sin memoria para salida\n"); return; }

    int W = in->width, H = in->height, C = in->unidadxpixel;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (C == 1) {
                float acc = 0.f, wsum = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            float w = K[ky + 1][kx + 1];
                            acc += in->data[IDX(in, nx, ny)] * w;
                            wsum += w;
                        }
                    }
                }
                float val = (normalize && wsum > 0.f) ? (acc / wsum) : acc;
                out->data[IDX(out, x, y)] = clampi((int)(val + 0.5f), 0, out->max_color);
            }
            else { // C==3
                float ar = 0.f, ag = 0.f, ab = 0.f, wsum = 0.f;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = x + kx, ny = y + ky;
                        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                            float w = K[ky + 1][kx + 1];
                            int p = IDX(in, nx, ny);
                            ar += in->data[p + 0] * w;
                            ag += in->data[p + 1] * w;
                            ab += in->data[p + 2] * w;
                            wsum += w;
                        }
                    }
                }
                if (normalize && wsum > 0.f) { ar /= wsum; ag /= wsum; ab /= wsum; }
                int o = IDX(out, x, y);
                out->data[o + 0] = clampi((int)(ar + 0.5f), 0, out->max_color);
                out->data[o + 1] = clampi((int)(ag + 0.5f), 0, out->max_color);
                out->data[o + 2] = clampi((int)(ab + 0.5f), 0, out->max_color);
            }
        }
    }
}

void usage(const char* p) {
    std::fprintf(stderr, "Uso:\n  %s <input.pgm|input.ppm>\n", p);
}

int main(int argc, char* argv[]) {
    std::clock_t c0 = std::clock();     // tiempo de CPU
    std::time_t  w0 = std::time(NULL);  // wall-clock (segundos)
    if (argc < 2) { usage(argv[0]); return 1; }
    const char* in_path = argv[1];

    const float K_blur[3][3] = { {1.f / 9,1.f / 9,1.f / 9},{1.f / 9,1.f / 9,1.f / 9},{1.f / 9,1.f / 9,1.f / 9} };
    const float K_laplace[3][3] = { {0,-1,0},{-1,4,-1},{0,-1,0} };
    const float K_sharpen[3][3] = { {0,-1,0},{-1,5,-1},{0,-1,0} };

    Image in = {};
    if (!read_image(in_path, &in)) return 1;

    char base[256]; base_from_path(in_path, base, sizeof(base));

    Image out_blur = {}, out_lap = {}, out_shp = {};
    ensure_output_dir();

    // Tres filtros en paralelo (cada sección ejecuta el barrido 3x3 completo).
#pragma omp parallel sections
    {
#pragma omp section
        { apply_kernel3x3(&in, &out_blur, K_blur, 1); }

#pragma omp section
        { apply_kernel3x3(&in, &out_lap, K_laplace, 0); }

#pragma omp section
        { apply_kernel3x3(&in, &out_shp, K_sharpen, 0); }
    }

    char p_blur[MAX_PATH_LEN], p_lap[MAX_PATH_LEN], p_shp[MAX_PATH_LEN];
    build_named(p_blur, sizeof(p_blur), base, "blur", &in);
    build_named(p_lap, sizeof(p_lap), base, "laplace", &in);
    build_named(p_shp, sizeof(p_shp), base, "sharpen", &in);
    std::clock_t c1 = std::clock();
    std::time_t  w1 = std::time(NULL);
    double cpu_ms = 1000.0 * (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
    double wall_s = std::difftime(w1, w0); // en segundos

    std::printf("Tiempo CPU: %.3f ms | Tiempo total: %.0f s\n", cpu_ms, wall_s);
    if (!write_image(p_blur, &out_blur)) std::fprintf(stderr, "No se pudo guardar %s\n", p_blur);
    if (!write_image(p_lap, &out_lap)) std::fprintf(stderr, "No se pudo guardar %s\n", p_lap);
    if (!write_image(p_shp, &out_shp)) std::fprintf(stderr, "No se pudo guardar %s\n", p_shp);

    std::printf("Listo:\n  %s\n  %s\n  %s\n", p_blur, p_lap, p_shp);

    std::free(in.data);
    std::free(out_blur.data);
    std::free(out_lap.data);
    std::free(out_shp.data);
    return 0;
}