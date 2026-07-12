# Analisis Arsitektur & Panduan Implementasi Multi-Repo Laravel Menggunakan Subfolder Symlink
**Studi Kasus: Sistem Terintegrasi `gii.smartid.co.id`**

Dokumen ini menyajikan analisis arsitektur, komparasi keputusan, serta panduan teknis implementasi untuk pengembangan dan deployment sistem terintegrasi **GII SmartID**. 

Sistem ini terbagi menjadi beberapa modul Laravel independen yang dikembangkan oleh tim terpisah, namun harus diakses di bawah satu subdomain utama tanpa menggunakan container/Docker karena batasan resource server.

---

## 1. Latar Belakang & Kebutuhan Bisnis

Aplikasi **GII SmartID** terdiri dari empat komponen utama yang semuanya dibangun menggunakan framework Laravel:
1.  **Landing Page**: Halaman pemasaran utama yang memiliki frekuensi perubahan (perbaikan tampilan, promo, copywriting) sangat tinggi.
2.  **Dashboard System**: Core system/CMS utama untuk memproses bisnis internal.
3.  **SSO (Single Sign-On)**: Gerbang autentikasi terpusat.
4.  **Payment System**: Sistem transaksi keuangan yang membutuhkan stabilitas dan keamanan tingkat tinggi.

### Dilema Pengembangan:
*   Landing page sangat sering diperbarui. Jika digabungkan dalam satu kode program (Monolitik/1 Repo) dengan sistem Payment dan SSO, setiap kali ada perubahan kecil pada landing page, seluruh sistem harus dideploy ulang. Hal ini meningkatkan risiko terjadinya bug fatal pada modul Payment dan SSO yang sensitif.
*   Tim pengembang ingin memisahkan modul-modul ini agar pengerjaan lebih fokus, aman, dan deployment bisa dilakukan secara mandiri (*independent deployment*).

---

## 2. Analisis & Perbandingan Solusi Arsitektur

Tim sempat memperdebatkan dua pendekatan ekstrem sebelum akhirnya merumuskan solusi alternatif ketiga:

| Kriteria Analisis | Pilihan A: Monorepo (1 Repositori) | Pilihan B: Container (Docker/Kubernetes) | Pilihan C (Usulan): Multi-Repo + Symlink Subfolder |
| :--- | :--- | :--- | :--- |
| **Isolasi Kode** | 🔴 Buruk (Semua developer memiliki akses ke kode sensitif SSO & Payment). | 🟢 Sangat Baik (Isolasi total tingkat sistem operasi dan kode). | 🟢 Sangat Baik (Repo terpisah, developer bekerja hanya pada modulnya). |
| **Isolasi Database & Migrasi** | 🔴 Buruk (Rawan tabrakan file migrasi saat merge, database tergabung jadi satu). | 🟢 Sangat Baik (Masing-masing modul memiliki database terisolasi). | 🟢 Sangat Baik (Dapat dipisah ke database berbeda, migrasi aman di repo masing-masing). |
| **Resource Server** | 🟢 Sangat Ringan (Berjalan di atas 1 instance server). | 🔴 Sangat Berat (Setiap modul butuh container OS virtual. Butuh RAM minimal 4-8GB). | 🟢 Sangat Ringan (Berjalan langsung di web server & PHP bawaan OS. RAM 2GB sudah cukup). |
| **Kompleksitas Infrastruktur** | 🟢 Sangat Rendah (Setup hosting/VPS standar). | 🔴 Sangat Tinggi (Memerlukan keahlian DevOps untuk Docker, CI/CD, dan orkestrasi). | 🟡 Sedang (Memerlukan setup `.htaccess` dan symlink di awal, setelah itu berjalan otomatis). |
| **Kemudahan Deployment** | 🟡 Sedang (Deployment lambat karena harus memproses seluruh kode monolitik). | 🟢 Sangat Baik (Deploy per container secara independen). | 🟢 Sangat Baik (Hanya deploy folder modul yang berubah via `git pull` per direktori). |

### Kesimpulan Keputusan:
**Pilihan C (Multi-Repo dengan Symlink Subfolder)** dipilih sebagai solusi paling rasional. Solusi ini memberikan keuntungan isolasi kode dan database layaknya *Microservices*, namun tetap mempertahankan efisiensi penggunaan memori (RAM) dan kemudahan manajemen server layaknya *Monolith*.

---

## 3. Desain Arsitektur Folder & Routing (`gii.smartid.co.id`)

Agar tidak bentrok secara routing, modul-modul tersebut dipetakan ke dalam path URL berikut:
*   **Landing Page**: `https://gii.smartid.co.id/`
*   **Dashboard System**: `https://gii.smartid.co.id/dashboard`
*   **SSO System**: `https://gii.smartid.co.id/sso`
*   **Payment System**: `https://gii.smartid.co.id/payment`

### Struktur Direktori Server (VPS / Shared Hosting):
Kode program utama (core engine) diletakkan di luar folder publik demi alasan keamanan, lalu dihubungkan ke folder public subdomain utama menggunakan Symbolic Link (Symlink).

```
/home/username/
├── project_sources/
│   ├── gii-landing/                 <-- Repo Git Landing Page
│   │   ├── app/
│   │   └── public/                  <-- Document root asli landing page
│   ├── gii-dashboard/               <-- Repo Git Dashboard System
│   │   ├── app/
│   │   └── public/                  <-- Berisi index.php dashboard
│   ├── gii-sso/                     <-- Repo Git SSO
│   │   └── public/
│   └── gii-payment/                 <-- Repo Git Payment
│       └── public/
│
└── domains/
    └── smartid.co.id/
        └── public_html/
            └── gii/                  <-- Folder Subdomain (gii.smartid.co.id)
                ├── [File publik dari gii-landing/public, e.g. index.php utama]
                ├── .htaccess        <-- HTAccess Utama (Landing Page)
                ├── dashboard        <-- Symlink ke project_sources/gii-dashboard/public
                ├── sso              <-- Symlink ke project_sources/gii-sso/public
                └── payment          <-- Symlink ke project_sources/gii-payment/public
```

---

## 4. Panduan Teknis Konfigurasi Server (Kunci Stabilitas)

Konfigurasi `.htaccess` sangat krusial untuk mencegah web server LiteSpeed/Apache memperlakukan symlink subfolder sebagai direktori fisik biasa (yang memicu error **403 Forbidden**) atau salah meneruskan route ke aplikasi landing page (memicu error **404 Not Found**).

### Langkah 1: Setup Symlink di Server
Jalankan perintah ini di SSH server untuk membuat jembatan link:
```bash
ln -s ~/project_sources/gii-dashboard/public ~/domains/smartid.co.id/public_html/gii/dashboard
ln -s ~/project_sources/gii-sso/public ~/domains/smartid.co.id/public_html/gii/sso
ln -s ~/project_sources/gii-payment/public ~/domains/smartid.co.id/public_html/gii/payment
```

### Langkah 2: Konfigurasi `.htaccess` Root Subdomain (`public_html/gii/.htaccess`)
File `.htaccess` utama ini berfungsi untuk mengatur Landing Page sekaligus bertindak sebagai pengarah jalur (*traffic controller*) agar request subfolder langsung dialihkan ke index.php masing-masing modul.

```apache
DirectoryIndex index.php

<IfModule mod_rewrite.c>
    RewriteEngine On
    RewriteBase /

    # ==========================================
    # 1. BYPASS ROUTING UNTUK SUBFOLDER SYMLINK
    # ==========================================
    
    # Teruskan request /dashboard ke file index.php miliknya sendiri
    RewriteCond %{REQUEST_URI} ^/dashboard [NC]
    RewriteRule ^dashboard/(.*)$ dashboard/index.php [L]

    # Teruskan request /sso ke file index.php miliknya sendiri
    RewriteCond %{REQUEST_URI} ^/sso [NC]
    RewriteRule ^sso/(.*)$ sso/index.php [L]

    # Teruskan request /payment ke file index.php miliknya sendiri
    RewriteCond %{REQUEST_URI} ^/payment [NC]
    RewriteRule ^payment/(.*)$ payment/index.php [L]

    # ==========================================
    # 2. ROUTING STANDAR UNTUK LANDING PAGE
    # ==========================================
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteRule ^ index.php [L]
</IfModule>
```

### Langkah 3: Konfigurasi `.htaccess` pada Modul Subfolder (Contoh: Dashboard)
Edit file `.htaccess` di dalam folder public modul masing-masing (misal: `project_sources/gii-dashboard/public/.htaccess`):

```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Set base path sesuai dengan nama subfolder URL
    RewriteBase /dashboard/

    # Redirect Trailing Slashes...
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteCond %{REQUEST_URI} (.+)/$
    RewriteRule ^ %1 [L,R=301]

    # Handle Front Controller...
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteRule ^ index.php [L]
</IfModule>
```
*(Lakukan hal yang sama untuk modul `sso` dan `payment` dengan menyesuaikan nilai `RewriteBase /sso/` dan `RewriteBase /payment/`).*

---

## 5. Standardisasi Penulisan Kode (Wajib Diikuti Developer)

Semua tim pengembang modul wajib menaati aturan penulisan kode ini untuk mencegah terjadinya link rusak di server produksi.

### A. Konfigurasi Isolasi File `.env`
Sesuaikan `.env` masing-masing proyek secara spesifik:
*   **Proyek Dashboard (`gii-dashboard`)**:
    ```env
    APP_URL=https://gii.smartid.co.id/dashboard
    SESSION_COOKIE=gii_dashboard_session
    SESSION_PATH=/dashboard
    ```
*   **Proyek SSO (`gii-sso`)**:
    ```env
    APP_URL=https://gii.smartid.co.id/sso
    SESSION_COOKIE=gii_sso_session
    SESSION_PATH=/sso
    ```

### B. Larangan Menulis Hardcoded URL Path
Developer dilarang keras menuliskan path absolut secara langsung dalam kode (HTML/Blade maupun Controller):
*   🔴 **SALAH**: `<a href="/login">Masuk</a>` (akan mengarah ke `gii.smartid.co.id/login` / landing page).
*   🟢 **BENAR**: `<a href="{{ route('login') }}">Masuk</a>` (Laravel secara otomatis mengenerate `gii.smartid.co.id/dashboard/login` mengikuti konfigurasi `APP_URL`).
*   🔴 **SALAH**: `return redirect('/home');`
*   🟢 **BENAR**: `return redirect()->route('home');`

### C. Pengaturan Middleware Redirect (Laravel 11 ke atas)
Pada modul dashboard, SSO, dan payment, ubah cara redirect default middleware guest di file `bootstrap/app.php` agar menggunakan route name:
```php
$middleware->redirectTo(fn () => route('login'));
```

### D. Konfigurasi Endpoint Livewire (Jika menggunakan Livewire/Filament)
Daftarkan override route di `app/Providers/AppServiceProvider.php` masing-masing modul subfolder agar request AJAX Livewire tidak terkena error 404:
```php
public function register(): void
{
    if (class_exists(\Livewire\Livewire::class)) {
        \Livewire\Livewire::setUpdateRoute(function ($handle) {
            return Route::post('/dashboard/livewire/update', $handle); // Sesuaikan prefix subfolder
        });
        \Livewire\Livewire::setScriptRoute(function ($handle) {
            return Route::get('/dashboard/livewire/livewire.js', $handle); // Sesuaikan prefix subfolder
        });
    }
}
```

---

## 6. Mekanisme Integrasi SSO & Database Sharing

Untuk mewujudkan *Single Sign-On* di mana pengguna yang masuk melalui modul SSO otomatis terautentikasi di modul Dashboard dan Payment tanpa harus login kembali:

1.  **Shared Session Domain**: Di file `.env` semua modul, atur domain session yang sama:
    ```env
    SESSION_DOMAIN=gii.smartid.co.id
    ```
2.  **Shared Session Driver**: Gunakan basis data atau Redis terpusat yang sama sebagai penyimpan data session. Pengaturan di `.env` semua modul:
    ```env
    SESSION_DRIVER=database
    # Hubungkan koneksi database session ke DB yang sama
    ```
3.  **APP_KEY yang Selaras**: Pastikan enkripsi session cookie menggunakan kunci yang kompatibel atau proses dekripsi token user dialihkan melalui middleware terpusat milik modul SSO.

---

## 7. Alur Kerja Pembaruan Kode & Deployment Rutin

### ⚠️ PERINGATAN ROUTE CACHE (PENTING!):
Karena file route Laravel (`web.php`) terkadang mengandung closure (misal fungsi callback instan), **JANGAN SEKALI-KALI** menjalankan perintah `php artisan route:cache` di lingkungan subfolder. Menjalankan cache route pada konfigurasi ini akan memicu error **405 Method Not Allowed** atau **404 Not Found**.

Gunakan perintah ini setiap kali melakukan pembaruan kode di server:
```bash
# 1. Pindah ke modul yang diperbarui
cd ~/project_sources/gii-dashboard

# 2. Tarik kode terbaru
git pull origin main

# 3. Bersihkan cache secara aman
php artisan config:cache
php artisan route:clear      <-- Gunakan CLEAR, bukan CACHE untuk routing
php artisan view:clear
php artisan view:cache
php artisan migrate --force
```

---
*Dokumen Analisis Arsitektur GII SmartID - Dipersiapkan untuk Tim Developer.*
