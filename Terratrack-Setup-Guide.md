# Panduan Setup & Deployment Terratrack di Server Hostinger

Panduan ini menjelaskan langkah-langkah untuk melakukan deployment **Terratrack** pada server production Hostinger dengan domain **`https://itsar.futurehero.id/teratrack`**.

Untuk menjaga keamanan core engine Laravel, kita akan meletakkan file source code di luar direktori publik (`project_sources`) dan membuat tautan simbolik (symlink) dari `public_html`.

---

## Langkah 1: Konek ke Server via SSH
Sebelum melakukan operasi Git di server, Anda harus masuk ke terminal server Hostinger.

1. Buka dashboard Hostinger, cari menu **Advanced > SSH Access**.
2. Pastikan status SSH adalah **Enabled**.
3. Salin perintah **SSH Command** yang disediakan (contoh: `ssh username@ip_address -p port`).
4. Buka Terminal (Mac/Linux) atau Git Bash/PowerShell (Windows), paste perintah tersebut, lalu masukkan password SSH hosting Anda.

---

## Langkah 2: Setup SSH Key di Server (Akses ke GitHub)
Agar server Hostinger Anda bisa melakukan `git pull` dari private repository GitHub tanpa terus-menerus meminta password, daftarkan SSH Key server ke GitHub.

1. **Generate SSH Key baru di server:**
   Jalankan perintah ini di dalam terminal SSH Hostinger (tekan Enter saja jika ada pertanyaan/passphrase):
   ```bash
   ssh-keygen -t ed25519 -C "server-hostinger"
   ```
2. **Ambil public key yang baru dibuat:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
3. Salin seluruh teks yang muncul (dimulai dari `ssh-ed25519 ...`).
4. **Daftarkan ke GitHub:**
   * Buka GitHub repository Terratrack Anda (`https://github.com/itsr11/TerraTrack`).
   * Pergi ke **Settings > Deploy keys > Add deploy key**.
   * Paste key tadi ke kolom **Key**, beri nama (misal: *Hostinger Production*), dan klik **Add key**.

---

## Langkah 3: Clone / Pull Pertama Kali ke `project_sources`
Kita akan meletakkan core engine Terratrack di luar direktori `public_html` agar file sistem Laravel tidak dapat diakses langsung oleh publik.

1. **Masuk ke direktori domain Anda:**
   Biasanya di Hostinger strukturnya adalah `domains/futurehero.id/`. Buat folder `project_sources` di sana:
   ```bash
   cd ~/domains/futurehero.id/
   mkdir -p project_sources
   cd project_sources
   ```
2. **Clone repository Terratrack menggunakan SSH URL:**
   ```bash
   git clone git@github.com:itsr11/TerraTrack.git terratrack
   ```

---

## Langkah 4: Setup Environment & Symlink Publik
Kita perlu menghubungkan folder `public` milik Terratrack ke folder `public_html/itsar/teratrack` agar dapat diakses oleh browser.

1. **Setup File `.env` Produksi:**
   ```bash
   cd ~/domains/futurehero.id/project_sources/teratrack
   cp .env.example .env
   nano .env
   ```
   Sesuaikan konfigurasi berikut di dalam `.env`:
   ```env
   APP_NAME=Terratrack
   APP_ENV=production
   APP_DEBUG=false
   APP_URL=https://itsar.futurehero.id/teratrack

   DB_CONNECTION=sqlite
   
   # Isolasi Cookie Sesi agar tidak konflik di subdomain
   SESSION_COOKIE=terratrack_session
   SESSION_PATH=/terratrack
   ```
   *Tekan `CTRL+X`, lalu `Y`, lalu `Enter` untuk menyimpan.*

2. **Install Dependencies & Generate Key:**
   ```bash
   # Pasang package production
   composer install --no-dev --optimize-autoloader
   
   # Hasilkan security key
   php artisan key:generate
   
   # Buat file database SQLite kosong
   touch database/database.sqlite
   
   # Jalankan migrasi dan seeder database
   php artisan migrate --seed --force
   ```

3. **Hubungkan ke folder publik (`public_html`):**
   Gunakan Symlink agar folder `public_html/itsar/teratrack` langsung membaca folder `public` di core source code kita:
   ```bash
   # Pastikan folder tujuan symlink belum dibuat sebelumnya
   rm -rf ~/domains/futurehero.id/public_html/itsar/teratrack
   
   # Jalankan pembuatan symlink
   ln -s ~/domains/futurehero.id/project_sources/teratrack/public ~/domains/futurehero.id/public_html/itsar/teratrack
   ```

4. **Set Izin Akses SQLite (Kritis):**
   Agar database SQLite dapat ditulis oleh web server saat ada transaksi presence check-in & quiz:
   ```bash
   chmod -R 775 database
   chmod 664 database/database.sqlite
   
   # Berikan hak akses write ke folder storage & cache
   chmod -R 775 storage bootstrap/cache
   ```

---

## Langkah 5: Alur Kerja Setiap Kali Ada Update (Workflow Rutin)
Setiap kali Anda melakukan perubahan kode di komputer lokal, lakukan `git push` ke GitHub seperti biasa. Untuk memperbarui kode di server production Hostinger, jalankan perintah berikut via SSH:

```bash
# 1. Masuk ke folder project
cd ~/domains/futurehero.id/project_sources/terratrack

# 2. Tarik kode terbaru dari GitHub
git pull origin main

# 3. Jalankan migrasi dan optimasi cache Laravel
php artisan config:cache
php artisan route:cache
php artisan view:cache
php artisan migrate --force

# 4. Pastikan permission database tetap aman
chmod 664 database/database.sqlite
```
