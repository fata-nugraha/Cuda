# Petunjuk Penggunaan Program
- Pada folder yang mengandung `Makefile`, Ketik `make` pada terminal.
- Program akan dikompilasi.
- Setelah itu, jalankan program dengan perintah `./prog <N>` pada direktori yang terdapat Makefile.
- Contoh: `./prog 100` akan menjalankan program dengan N (jumlah node) = 100.
- Akan keluar file output dengan nama `output-<N>.txt` pada direktori yang terdapat Makefile.

# Pembagian Tugas
- Fata Nugraha (13517109) mengerjakan fungsi dijkstra yang berjalan pada gpu dan main program.
- Edward Alexander jaya (13517115) mengerjakan fungsi dijkstra yang berjalan pada gpu dan fungsi file eksternal.

# Laporan Pengerjaan
#### Deskripsi Solusi Paralel
- Terdapat gridDim dan blockDim pada program kami. Perhatikan bahwa 1 grid mempunyai block sebanyak gridDim dan 1 block mempunyai thread sebanyak blockDim.
- gridDim yang dipakai adalah sejumlah N (jumlah node) dan blockDim yang dipakai adalah sejumlah 1 untuk menjalankan algoritma dijkstra. Artinya, terdapat N block yang menjalankan dijkstra. Setiap block mengeksekusi dijkstra dari sebuah node ke semua node lain.

#### Analisis Solusi Paralel

Alokasikan semua array 1D yang dibutuhkan pada CPU dan GPU:

    cudaMallocManaged((void **) &dijkstraGraph, (sizeof(int) * N * N));
    cudaMallocManaged((void **) &result, (sizeof(int) * N * N));

Kemudian lakukan alokasi array 1D gpuVisited pada GPU untuk menentukan node yang sudah dikunjungi:

    cudaMalloc((void **) &gpuVisited, (sizeof(bool) * N * N));

Setelah itu, panggil fungsi dijkstra dengan gridDim = N dan blockDim = 1:

    dijkstra<<<N, 1>>>(dijkstraGraph, result, gpuVisited, N);

Karena blockDim = 1, maka representasi index adalah

      int blockIndex1D = N * blockIdx.x;

Pertama, array visited diinisiasi dengan false dan array result diinisiasi dengan INT_MAX, kecuali pada indeks N * blockIdx.x + blockIdx.x:

      for (int vertex = 0; vertex < N; vertex++) {
          visited[blockIndex1D + vertex] = false;
          result[blockIndex1D + vertex] = INT_MAX;
      }

      // Distance from source to itself = 0
      result[blockIndex1D + blockIdx.x] = 0;

Kemudian lakukan algoritma dijkstra dan akan didapatkan array 1D result yang baru.
Perhatikan bahwa hasil dari dijkstra disimpan ke dalam array 1D result pada fungsi:

      __global__ void dijkstra(int *graph, int *result, bool* visited, int N)



#### Jumlah Thread yang Digunakan
- Ada N buah thread yang digunakan, yang tersebar dalam N buah block dengan 1 buah thread di masing-masing blocknya. Alasannya, setiap thread tidak perlu saling mengetahui keadaan di thread lain, sehingga thread dapat dipisahkan pada block-block yang berbeda.

#### Pengukuran Kinerja untuk tiap Kasus Uji
Berikut adalah hasil pengujian yang dikerjakan pada server 167.205.32.100:
- **N = 100**

  | Tipe | Percobaan 1 | Percobaan 2 | Percobaan 3 |
  |---|--- |---|---|
  | Serial   | 19318.000000 µs  | 23879.000000 µs   | 21159.000000 µs|
  | Paralel | 23200.000000 µs | 23051.000000 µs | 23130.000000 µs|

- **N = 500**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial |  1525218.000000 µs |  1505172.000000 µs |  1461822.000000 µs |
  | Paralel  |  451472.000000 µs |  451586.000000 µs |  451458.000000 µs |
- **N = 1000**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial | 10383358.000000 µs | 10314490.000000 µs | 10425789.000000 µs |
  | Paralel  | 4756026.000000 µs |  4764137.000000 µs | 4749206.000000 µs |
- **N = 3000**

  | Tipe  |  Percobaan 1 | Percobaan 2  | Percobaan 3  |
  |---|---|---|---|
  | Serial | 300571112.000000 µs  |  300935149.000000 µs |  306387030.000000 µs|
  | Paralel  | 113458008.000000 µs | 113471208.000000 µs |  113479976.000000 µs|

#### Analisis Perbandingan Kinerja Serial dan Paralel
- Pada program serial, hanya ada satu proses yang menjalankan program dijkstra. Pada program paralel dengan GPU, program dapat dijalankan pada N proses yang berbeda, tetapi ada overhead berupa waktu transfer data dari CPU ke GPU.
- Untuk program dengan N kecil, waktu eksekusi program secara serial dan waktu eksekusi program secara paralel hampir sama. Proses di GPU tentu lebih cepat dari CPU, karena ada N buah pekerjaan yang dilakukan bersamaan, tetapi overhead transfer data dari CPU ke GPU membuat waktu eksekusi totalnya hampir sama.
- Untuk program dengan N besar, waktu eksekusinya di GPU jauh lebih cepat daripada waktu eksekusi di CPU. Alasannya, waktu transfer data kira-kira berderajat O(n) sedangkan waktu eksekusi dijkstra untuk semua node berderajat O(n^3), sehingga overhead waktu transfer data ke GPU lebih kecil daripada selisih waktu eksekusi di CPU dan GPU.