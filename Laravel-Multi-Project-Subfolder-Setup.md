# Panduan Teknis: Setup & Deployment Laravel Multi-Project dalam Subfolder Subdomain

Panduan ini ditujukan bagi developer untuk melakukan setup, integrasi, dan deployment beberapa proyek Laravel mandiri di bawah satu subdomain yang sama menggunakan metode **Subfolder Symlink** (misalnya `https://itsar.futurehero.id/nama-proyek`) di server Hostinger dengan web server LiteSpeed.

---

## 1. Persyaratan Sistem & Server

Sebelum memulai, pastikan server memenuhi kriteria berikut:
1. **Akses SSH**: Wajib memiliki akses SSH ke server Hostinger untuk melakukan symlink, git operations, dan artisan commands.
2. **Web Server**: LiteSpeed / Apache (mendukung pembacaan file `.htaccess`).
3. **Versi PHP**: PHP 8.2 ke atas (sesuaikan dengan kebutuhan framework).
4. **Git**: Terpasang di server untuk mempermudah pembaruan kode.

---

## 2. Arsitektur Direktori yang Direkomendasikan (Best Practice)

Untuk menjaga keamanan kode program utama (core engine), **sangat dilarang** meletakkan seluruh file proyek langsung di dalam folder publik `public_html`. Gunakan pemisahan folder berikut:

```
/home/username/
├── project_sources/                 <-- Folder khusus untuk source code core proyek
│   ├── vetprep-ai/                  <-- Core Code Proyek A (Git repository)
│   │   ├── app/
│   │   ├── bootstrap/
│   │   ├── public/                  <-- Folder public asli (berisi index.php)
│   │   └── ...
│   └── proyek-lain/                 <-- Core Code Proyek B
│
└── domains/
    └── futurehero.id/
        └── public_html/
            └── itsar/               <-- Direktori Subdomain Utama (itsar.futurehero.id)
                ├── code.html        <-- Landing page portofolio utama
                ├── .htaccess        <-- Konfigurasi routing subdomain utama (PENTING!)
                ├── vetprep-ai       <-- Symlink ke project_sources/vetprep-ai/public
                └── proyek-lain      <-- Symlink ke project_sources/proyek-lain/public
```

---

## 3. Langkah-Langkah Deployment Proyek Baru

### Langkah 1: Clone Source Code ke `project_sources`
Masuk ke terminal server via SSH, kemudian navigasi ke folder source code dan lakukan clone:
```bash
mkdir -p ~/domains/futurehero.id/project_sources
cd ~/domains/futurehero.id/project_sources
git clone <repository_url> nama-proyek
```

### Langkah 2: Konfigurasi File `.env`
Salin template `.env` dan konfigurasikan secara spesifik agar tidak bentrok dengan proyek lain di subdomain yang sama:
```bash
cd nama-proyek
cp .env.example .env
nano .env
```

**Konfigurasi Kunci di `.env`:**
```env
APP_NAME="Nama Proyek"
APP_ENV=production
APP_DEBUG=false

# 1. Gunakan URL lengkap beserta subfoldernya
APP_URL=https://itsar.futurehero.id/nama-proyek

# 2. Isolasi Session Cookie agar tidak bertabrakan dengan proyek lain
SESSION_COOKIE=namaproyek_session
SESSION_PATH=/nama-proyek
```

### Langkah 3: Instalasi Dependensi & Setup
Jalankan Composer untuk menginstal package tanpa menyertakan dev-tools demi keamanan dan kecepatan:
```bash
composer install --no-dev --optimize-autoloader
php artisan key:generate
```

### Langkah 4: Hubungkan ke Publik dengan Symlink
Buat tautan simbolik (symlink) dari folder public proyek ke folder subdomain utama:
```bash
# Hapus folder/file duplikat di folder subdomain jika ada
rm -rf ~/domains/futurehero.id/public_html/itsar/nama-proyek

# Buat symlink baru
ln -s ~/domains/futurehero.id/project_sources/nama-proyek/public ~/domains/futurehero.id/public_html/itsar/nama-proyek
```

---

## 4. Konfigurasi Server & Routing Penting (Crucial Fixes)

Menjalankan Laravel di dalam subfolder melalui symlink sering kali menimbulkan masalah **404 Not Found**, **403 Forbidden**, atau **405 Method Not Allowed**. Pastikan konfigurasi berikut diterapkan:

### A. Versi PHP Khusus di Subfolder (`.htaccess` Proyek)
Jika server Anda menggunakan versi PHP default yang berbeda dengan kebutuhan proyek baru (misal proyek butuh PHP 8.4 sedangkan default server adalah PHP 8.1), tambahkan handler versi PHP di bagian paling atas file `project_sources/nama-proyek/public/.htaccess`:

```apache
# Force PHP 8.4 Wildcard Handler untuk Hostinger
<FilesMatch "\.(php4|php5|php3|php2|php|phtml)$">
    SetHandler application/x-httpd-alt-php84___wildcard
</FilesMatch>
```

### B. Konfigurasi Base URL Rewrite di Subfolder (`.htaccess` Proyek)
Edit file `project_sources/nama-proyek/public/.htaccess` agar mengenali base path subfolder:
```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Sesuaikan Base Path dengan nama subfolder/symlink Anda
    RewriteBase /nama-proyek/

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

### C. Bypass Directory Listing & 403 Forbidden (`.htaccess` Subdomain Utama)
LiteSpeed memiliki quirk di mana ia akan membaca URL subfolder (seperti `/nama-proyek` atau `/nama-proyek/admin`) sebagai folder fisik dan mencoba melakukan directory listing yang berujung pada error **403 Forbidden**. 

Untuk mengatasinya, tambahkan rewrite rule di file `.htaccess` subdomain utama (`public_html/itsar/.htaccess`) agar request tersebut langsung diteruskan ke index.php Laravel:

```apache
DirectoryIndex code.html index.php

RewriteEngine On

# Mengarahkan request root subfolder ke controller Laravel
RewriteRule ^nama-proyek/$ nama-proyek/index.php [L]

# Mengarahkan request admin panel Filament ke controller Laravel
RewriteRule ^nama-proyek/admin/?$ nama-proyek/index.php [L]
```

---

## 5. Modifikasi Kode Laravel (Code-Level Adjustment)

### A. Gunakan Named Routes untuk Semua Link & Redirect
**PANTANGAN BESAR**: Jangan pernah menuliskan url/path secara hardcoded di file Blade maupun Controller (misal: `href="/login"` atau `redirect('/dashboard')`). Browser akan mengarahkan user ke root subdomain utama (`https://itsar.futurehero.id/login`) sehingga menghasilkan error **404 Not Found**.

* **Solusi**: Gunakan Laravel `route()` helper yang secara otomatis mendeteksi `APP_URL` dari subfolder.
```html
<!-- SALAH -->
<a href="/login">Masuk</a>

<!-- BENAR -->
<a href="{{ route('login') }}">Masuk</a>
```
```php
// SALAH
return redirect('/dashboard');

// BENAR
return redirect()->route('dashboard');
```

### B. Konfigurasi Redirect Default di Middleware
Pada Laravel 11/12/13+, konfigurasi default redirect untuk guest/auth diletakkan di `bootstrap/app.php`. Ubah konfigurasi tersebut agar menggunakan named route dinamis:

```php
->withMiddleware(function (Middleware $middleware) {
    // Menggunakan route helper agar dinamis dengan prefix subfolder
    $middleware->redirectTo(fn () => route('login'));
})
```

### C. Penyesuaian Livewire Endpoint (Jika Menggunakan TALL Stack)
Tambahkan konfigurasi kustom di `app/Providers/AppServiceProvider.php` agar asset JS Livewire dan request AJAX-nya tidak menghasilkan error 404:

```php
public function register(): void
{
    if (class_exists(\Livewire\Livewire::class)) {
        \Livewire\Livewire::setUpdateRoute(function ($handle) {
            return Route::post('/nama-proyek/livewire/update', $handle);
        });
        \Livewire\Livewire::setScriptRoute(function ($handle) {
            return Route::get('/nama-proyek/livewire/livewire.js', $handle);
        });
    }
}
```

---

## 6. Alur Kerja Pembaruan Kode (Workflow Update)

### PENTING: Larangan Penggunaan Route Cache
Jika file routing Laravel Anda (`web.php` or `api.php`) memiliki Closure/fungsi callback (misal: `Route::post('/logout', function() { ... })`), **jangan sekali-kali menjalankan `php artisan route:cache`**. Serialisasi closure pada environment subfolder akan menyebabkan error **405 Method Not Allowed** atau **404 Not Found**.

Gunakan perintah pembersihan route berikut saat deployment rutin:

```bash
# Pindah ke direktori sumber proyek
cd ~/domains/futurehero.id/project_sources/nama-proyek

# Tarik update kode terbaru
git pull origin main

# Jalankan perintah update & bersihkan cache (JANGAN CACHE ROUTE!)
php artisan config:cache
php artisan route:clear      <-- Gunakan clear, JANGAN route:cache
php artisan view:clear
php artisan view:cache
php artisan migrate --force
```

---

*Panduan ini dibuat berdasarkan penyelesaian case troubleshooting deployment `vetprep-ai` di server Hostinger/LiteSpeed subdomain `itsar.futurehero.id`.*
