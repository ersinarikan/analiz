#!/bin/bash
# T.C. Aile ve Sosyal Hizmetler Bakanlığı
# WSANALIZ DEB Paketi Test Scripti

set -e

echo "🧪 WSANALIZ DEB Paketi Test Scripti"
echo "=================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test fonksiyonları
test_package_creation() {
    echo -e "${BLUE}📦 DEB paketi oluşturma testi...${NC}"
    
    if python3 create_deb_package.py; then
        echo -e "${GREEN}✅ DEB paketi başarıyla oluşturuldu${NC}"
        
        if [ -f "wsanaliz_1.0.0_all.deb" ]; then
            echo -e "${GREEN}✅ Paket dosyası mevcut${NC}"
            
            # Paket bilgilerini göster
            echo -e "${BLUE}📋 Paket bilgileri:${NC}"
            dpkg-deb --info wsanaliz_1.0.0_all.deb
            
            return 0
        else
            echo -e "${RED}❌ Paket dosyası bulunamadı${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ DEB paketi oluşturulamadı${NC}"
        return 1
    fi
}

test_package_contents() {
    echo -e "${BLUE}📂 Paket içeriği testi...${NC}"
    
    if [ ! -f "wsanaliz_1.0.0_all.deb" ]; then
        echo -e "${RED}❌ Paket dosyası bulunamadı${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📋 Paket içeriği:${NC}"
    dpkg-deb --contents wsanaliz_1.0.0_all.deb | head -20
    
    # Kritik dosyaların varlığını kontrol et
    critical_files=(
        "./opt/wsanaliz/app.py"
        "./etc/nginx/sites-available/wsanaliz"
        "./etc/systemd/system/wsanaliz.service"
        "./DEBIAN/control"
        "./DEBIAN/postinst"
        "./DEBIAN/prerm"
    )
    
    for file in "${critical_files[@]}"; do
        if dpkg-deb --contents wsanaliz_1.0.0_all.deb | grep -q "$file"; then
            echo -e "${GREEN}✅ $file mevcut${NC}"
        else
            echo -e "${RED}❌ $file eksik${NC}"
            return 1
        fi
    done
    
    return 0
}

test_virtual_install() {
    echo -e "${BLUE}🔧 Sanal kurulum testi...${NC}"
    
    # Geçici dizin oluştur
    temp_dir=$(mktemp -d)
    echo "Geçici dizin: $temp_dir"
    
    # Paketi geçici dizine çıkart
    dpkg-deb --extract wsanaliz_1.0.0_all.deb "$temp_dir"
    
    # Kritik dizinlerin varlığını kontrol et
    critical_dirs=(
        "opt/wsanaliz"
        "etc/nginx/sites-available"
        "etc/systemd/system"
        "DEBIAN"
    )
    
    for dir in "${critical_dirs[@]}"; do
        if [ -d "$temp_dir/$dir" ]; then
            echo -e "${GREEN}✅ Dizin mevcut: $dir${NC}"
        else
            echo -e "${RED}❌ Dizin eksik: $dir${NC}"
            rm -rf "$temp_dir"
            return 1
        fi
    done
    
    # Script izinlerini kontrol et
    if [ -x "$temp_dir/DEBIAN/postinst" ]; then
        echo -e "${GREEN}✅ postinst script çalıştırılabilir${NC}"
    else
        echo -e "${RED}❌ postinst script çalıştırılabilir değil${NC}"
    fi
    
    if [ -x "$temp_dir/DEBIAN/prerm" ]; then
        echo -e "${GREEN}✅ prerm script çalıştırılabilir${NC}"
    else
        echo -e "${RED}❌ prerm script çalıştırılabilir değil${NC}"
    fi
    
    # Temizlik
    rm -rf "$temp_dir"
    
    return 0
}

test_dependencies() {
    echo -e "${BLUE}🔗 Bağımlılık testi...${NC}"
    
    # Paket bağımlılıklarını kontrol et
    deps=$(dpkg-deb --field wsanaliz_1.0.0_all.deb Depends)
    echo -e "${BLUE}📋 Bağımlılıklar: $deps${NC}"
    
    # Kritik bağımlılıkları kontrol et
    critical_deps=("python3" "nginx" "supervisor")
    
    for dep in "${critical_deps[@]}"; do
        if echo "$deps" | grep -q "$dep"; then
            echo -e "${GREEN}✅ Bağımlılık mevcut: $dep${NC}"
        else
            echo -e "${YELLOW}⚠️ Bağımlılık eksik: $dep${NC}"
        fi
    done
    
    return 0
}

test_lintian() {
    echo -e "${BLUE}🔍 Lintian (paket kalitesi) testi...${NC}"
    
    if command -v lintian &> /dev/null; then
        echo "Lintian analizi çalıştırılıyor..."
        lintian wsanaliz_1.0.0_all.deb || true
    else
        echo -e "${YELLOW}⚠️ Lintian kurulu değil, kalite testi atlanıyor${NC}"
        echo "Kurmak için: sudo apt install lintian"
    fi
    
    return 0
}

generate_test_report() {
    echo -e "${BLUE}📊 Test raporu oluşturuluyor...${NC}"
    
    report_file="deb_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "WSANALIZ DEB Paketi Test Raporu"
        echo "================================"
        echo "Tarih: $(date)"
        echo "Sistem: $(uname -a)"
        echo ""
        
        echo "Paket Bilgileri:"
        dpkg-deb --info wsanaliz_1.0.0_all.deb
        echo ""
        
        echo "Paket Boyutu:"
        ls -lh wsanaliz_1.0.0_all.deb
        echo ""
        
        echo "Paket İçeriği (İlk 50 satır):"
        dpkg-deb --contents wsanaliz_1.0.0_all.deb | head -50
        
    } > "$report_file"
    
    echo -e "${GREEN}✅ Test raporu oluşturuldu: $report_file${NC}"
}

# Ana test fonksiyonu
run_all_tests() {
    echo -e "${BLUE}🚀 Tüm testler çalıştırılıyor...${NC}"
    echo ""
    
    local failed_tests=0
    
    # Test 1: Paket oluşturma
    if ! test_package_creation; then
        ((failed_tests++))
    fi
    echo ""
    
    # Test 2: Paket içeriği
    if ! test_package_contents; then
        ((failed_tests++))
    fi
    echo ""
    
    # Test 3: Sanal kurulum
    if ! test_virtual_install; then
        ((failed_tests++))
    fi
    echo ""
    
    # Test 4: Bağımlılıklar
    if ! test_dependencies; then
        ((failed_tests++))
    fi
    echo ""
    
    # Test 5: Lintian
    test_lintian
    echo ""
    
    # Test raporu
    generate_test_report
    echo ""
    
    # Sonuç
    if [ $failed_tests -eq 0 ]; then
        echo -e "${GREEN}🎉 Tüm testler başarılı! ($failed_tests hata)${NC}"
        echo -e "${GREEN}✅ DEB paketi production'a hazır${NC}"
        return 0
    else
        echo -e "${RED}❌ $failed_tests test başarısız${NC}"
        echo -e "${RED}🔧 Hataları düzeltin ve tekrar deneyin${NC}"
        return 1
    fi
}

# Kullanım bilgisi
show_usage() {
    echo "Kullanım: $0 [seçenek]"
    echo ""
    echo "Seçenekler:"
    echo "  all        - Tüm testleri çalıştır (varsayılan)"
    echo "  create     - Sadece paket oluşturma testi"
    echo "  contents   - Sadece paket içeriği testi"
    echo "  install    - Sadece sanal kurulum testi"
    echo "  deps       - Sadece bağımlılık testi"
    echo "  lintian    - Sadece lintian testi"
    echo "  report     - Sadece test raporu oluştur"
    echo "  help       - Bu yardım mesajını göster"
}

# Ana program
case "${1:-all}" in
    "all")
        run_all_tests
        ;;
    "create")
        test_package_creation
        ;;
    "contents")
        test_package_contents
        ;;
    "install")
        test_virtual_install
        ;;
    "deps")
        test_dependencies
        ;;
    "lintian")
        test_lintian
        ;;
    "report")
        generate_test_report
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Geçersiz seçenek: $1${NC}"
        show_usage
        exit 1
        ;;
esac 