# Panduan Setup & Deployment Proyek Baru di Subdomain `itsar.futurehero.id`

Panduan ini berisi template dan langkah-langkah untuk menambahkan proyek baru (misalnya Laravel, HTML, atau framework lainnya) di bawah URL **`https://itsar.futurehero.id/nama-proyek-baru`**.

---

## 1. Arsitektur Folder yang Direkomendasikan (Best Practice)

Untuk menjaga keamanan kode program utama (core engine), sangat disarankan memisahkan file inti aplikasi dari direktori yang dapat diakses publik (`public_html`).

### Struktur Folder di Server (`/home/u691050390/`):
```
/home/u691050390/
├── project_sources/
│   └── nama-proyek-baru/            <-- Core Code Aplikasi (Git repo / File Laravel utama)
│       ├── app/
│       ├── config/
│       ├── public/                  <-- Folder public asli (tempat index.php)
│       └── ...
│
└── domains/
    └── futurehero.id/
        └── public_html/
            └── itsar/               <-- Direktori Subdomain (itsar.futurehero.id)
                ├── code.html        <-- Landing Page Portofolio Utama
                ├── .htaccess        <-- HTAccess Subdomain Utama
                └── nama-proyek-baru <-- Symlink (Tautan) yang mengarah ke folder public aplikasi
```

---

## 2. Langkah-Langkah Deployment Baru

### Langkah 1: Hubungkan ke Server via SSH
Masuk ke terminal server Hostinger Anda menggunakan SSH:
```bash
ssh username@ip_address -p port
```

### Langkah 2: Clone atau Letakkan Source Code di `project_sources`
Masuk ke direktori sumber proyek dan buat folder baru jika belum ada:
```bash
cd ~/domains/futurehero.id/
mkdir -p project_sources
cd project_sources
```
Clone kode proyek Anda dari Git (atau upload file zip lalu ekstrak di sini):
```bash
git clone git@github.com:username/repository-name.git nama-proyek-baru
```

### Langkah 3: Konfigurasi File `.env` (Khusus Laravel/Framework Sejenis)
Masuk ke dalam proyek dan salin `.env.example` ke `.env`:
```bash
cd nama-proyek-baru
cp .env.example .env
nano .env
```
Sesuaikan variabel berikut untuk mencegah bentrokan domain:
```env
APP_NAME="Nama Proyek Baru"
APP_ENV=production
APP_DEBUG=false
# Pastikan URL lengkap dengan subfolder
APP_URL=https://itsar.futurehero.id/nama-proyek-baru

# Isolasi Session & Cookie agar tidak bentrok dengan portofolio utama atau proyek lain
SESSION_COOKIE=namaproyekbaru_session
SESSION_PATH=/nama-proyek-baru
```

### Langkah 4: Setup Dependensi & Environment
Jalankan kompilasi proyek:
```bash
# Install package production tanpa menyertakan dev-tools
composer install --no-dev --optimize-autoloader

# Buat aplikasi key baru
php artisan key:generate
```

### Langkah 5: Hubungkan Folder Publik Menggunakan Symlink (Simbolik Link)
Agar browser bisa mengakses aplikasi Anda melalui URL `itsar.futurehero.id/nama-proyek-baru`, kita perlu membuat Symlink dari folder public aplikasi ke dalam folder `public_html/itsar/`:

1. Pastikan tidak ada folder bernama `nama-proyek-baru` di dalam `public_html/itsar/`. Jika ada folder kosong, hapus terlebih dahulu:
   ```bash
   rm -rf ~/domains/futurehero.id/public_html/itsar/nama-proyek-baru
   ```
2. Jalankan perintah pembuatan link:
   ```bash
   ln -s ~/domains/futurehero.id/project_sources/nama-proyek-baru/public ~/domains/futurehero.id/public_html/itsar/nama-proyek-baru
   ```

---

## 3. Penyesuaian HTAccess & Routing Subfolder

### A. Konfigurasi `.htaccess` Proyek Baru
Pastikan file `.htaccess` di dalam direktori `nama-proyek-baru/public/.htaccess` mengizinkan rewrite path dengan base subfolder:

```apache
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Sesuaikan Base Path dengan nama subfolder Anda
    RewriteBase /nama-proyek-baru/

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

### B. Link Assets dinamis (Blade/HTML)
Selalu gunakan helper path absolut dinamis untuk meload CSS, JS, dan gambar:
* **Laravel Blade:** Gunakan `{{ asset('css/style.css') }}`. Jangan gunakan `/css/style.css` karena akan diarahkan ke root domain utama.
* **HTML Biasa:** Gunakan relative path tanpa slash di awal (misal: `css/style.css` bukan `/css/style.css`).

### C. Livewire Asset Endpoint (Jika Menggunakan TALL Stack)
Tambahkan baris berikut pada `app/Providers/AppServiceProvider.php` proyek baru Anda agar Livewire tidak menghasilkan error 404 ketika melakukan request AJAX:

```php
public function register(): void
{
    if (class_exists(\Livewire\Livewire::class)) {
        \Livewire\Livewire::setUpdateRoute(function ($handle) {
            return Route::post('/nama-proyek-baru/livewire/update', $handle);
        });
        \Livewire\Livewire::setScriptRoute(function ($handle) {
            return Route::get('/nama-proyek-baru/livewire/livewire.js', $handle);
        });
    }
}
```

---

## 4. Alur Kerja Update Rutin (Workflow)

Saat Anda memperbarui kode lokal di komputer Anda, lakukan push ke GitHub seperti biasa. Untuk memperbaruinya di server:

```bash
# 1. Pindah ke direktori core code proyek
cd ~/domains/futurehero.id/project_sources/nama-proyek-baru

# 2. Tarik kode terbaru dari GitHub
git pull origin main

# 3. Optimasi Cache Laravel
php artisan config:cache
php artisan route:cache
php artisan view:cache

# 4. Jalankan migrasi database jika ada perubahan struktur tabel
php artisan migrate --force
```
