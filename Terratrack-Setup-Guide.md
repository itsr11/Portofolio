# Panduan Setup Laravel Multi-Project: `itsar.futurehero.id/Terratrack`

Panduan ini menjelaskan arsitektur folder, konfigurasi `.env`, penyesuaian routing, dan settingan `.htaccess` untuk menjalankan project Laravel baru bernama **Terratrack** di dalam subfolder dari subdomain utama Anda.

---

## 1. Arsitektur Folder yang Aman (Best Practice)

Untuk menjaga keamanan, Anda tidak boleh meletakkan seluruh source code Laravel di dalam folder publik `public_html`. Kita akan memisahkan **Core Code** (di luar akses publik) dan **Public Assets** (di dalam akses publik).

### Struktur Folder pada Hosting (`/home/u691050390/`):
```
/home/u691050390/
├── project_sources/
│   └── Terratrack/                  <-- Core Code Laravel (git clone / upload di sini)
│       ├── app/
│       ├── bootstrap/
│       ├── config/
│       ├── database/
│       ├── ... (seluruh folder core)
│       └── public/                  <-- Folder public asli
│
└── domains/
    └── futurehero.id/
        └── public_html/
            └── itsar/               <-- Subdomain itsar.futurehero.id
                ├── code.html        <-- Landing Page Portofolio Utama
                ├── .htaccess        <-- HTAccess Subdomain Utama
                └── Terratrack/      <-- Jembatan Link Publik (Symlink ke core public)
```

---

## 2. Langkah Setup & Deployment

### Langkah 1: Upload / Clone Core Code
Upload zip project Laravel Anda atau lakukan `git clone` ke dalam folder `/home/u691050390/project_sources/Terratrack/`.

### Langkah 2: Buat Symlink Publik
Untuk menghubungkan folder publik subdomain dengan folder public core Laravel, jalankan perintah ini melalui terminal SSH:
```bash
ln -s /home/u691050390/project_sources/Terratrack/public /home/u691050390/domains/futurehero.id/public_html/itsar/Terratrack
```
*Catatan: Jika folder `Terratrack` kosong sudah terlanjur dibuat di dalam `public_html/itsar/`, hapus terlebih dahulu folder kosong tersebut sebelum menjalankan perintah symlink di atas.*

### Langkah 3: Konfigurasi File `.env` (`/project_sources/Terratrack/.env`)
Buka file `.env` di dalam folder core Terratrack, lalu sesuaikan konfigurasi berikut:

```env
# 1. Pastikan URL mengarah ke subfolder secara lengkap
APP_URL=https://itsar.futurehero.id/Terratrack

# 2. Isolasi Database
DB_DATABASE=u691050390_terratrack
DB_USERNAME=u691050390_terratrack_user
DB_PASSWORD=password_database_anda

# 3. Mencegah Konflik Session Cookie dengan Project Lain
SESSION_COOKIE=terratrack_session
```

---

## 3. Konfigurasi Server & Routing

### A. File `.htaccess` Project Terratrack
Buat atau edit file `.htaccess` di dalam `/home/u691050390/project_sources/Terratrack/public/.htaccess` (yang sekarang ter-link di `public_html/itsar/Terratrack/`):

```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Sesuaikan Base Path dengan nama subfolder
    RewriteBase /Terratrack/

    # Redirect Trailing Slashes If Not A Folder...
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteCond %{REQUEST_URI} (.+)/$
    RewriteRule ^ %1 [L,R=301]

    # Handle Front Controller...
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteRule ^ index.php [L]
</IfModule>
```

### B. Asset Routing dalam Blade Template
Pastikan penulisan link assets di template Laravel Blade Anda selalu menggunakan helper `asset()` agar dinamis mengarah ke subfolder:
```html
<!-- BENAR (Otomatis menghasilkan https://itsar.futurehero.id/Terratrack/css/app.css) -->
<link rel="stylesheet" href="{{ asset('css/app.css') }}">

<!-- SALAH (Akan menembak root domain: https://itsar.futurehero.id/css/app.css) -->
<link rel="stylesheet" href="/css/app.css">
```

### C. Konfigurasi Livewire (Jika menggunakan TALL Stack)
Agar request AJAX Livewire tidak menembak root domain utama (`itsar.futurehero.id/livewire/...`), daftarkan custom route Livewire di file `app/Providers/AppServiceProvider.php` project Terratrack Anda:

```php
<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Facades\Route;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        // Jalankan ini agar Livewire memproses request di subfolder /Terratrack/
        if (class_exists(\Livewire\Livewire::class)) {
            \Livewire\Livewire::setUpdateRoute(function ($handle) {
                return Route::post('/Terratrack/livewire/update', $handle);
            });
            \Livewire\Livewire::setScriptRoute(function ($handle) {
                return Route::get('/Terratrack/livewire/livewire.js', $handle);
            });
        }
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        //
    }
}
```

---

## 4. Perintah Tambahan di Server (SSH)

Setelah file terpasang dan database dikonfigurasi, jalankan perintah optimasi Laravel dari folder core (`/home/u691050390/project_sources/Terratrack/`):
```bash
# Pindah ke directory project core
cd /home/u691050390/project_sources/Terratrack

# Install dependensi PHP (jika belum dilakukan)
composer install --no-dev --optimize-autoloader

# Jalankan database migration
php artisan migrate --force

# Bersihkan dan optimalkan cache config/routing
php artisan config:cache
php artisan route:cache
php artisan view:cache
```
Panduan ini siap digunakan untuk mendokumentasikan deployment proyek baru Anda.
