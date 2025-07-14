# TRANSFER ÖĞRENME VE ARTIMSALI EĞİTİMİN YAŞA TAHMİNİ UYGULAMALARINDA BİRLEŞTİRİLMESİ: HİBRİT BİR YAKLAŞIM

## ÖZET

Bu çalışma, yaş tahmini problemine yönelik transfer öğrenme ve artımsal eğitim tekniklerinin sistematik birleştirilmesi için yenilikçi bir metodoloji önermektedir. Önerilen yaklaşım, InsightFace Buffalo modelini temel alarak, OpenCLIP tabanlı güven skorlaması ile desteklenen otomatik etiketleme (pseudo-labeling) mekanizması kullanmaktadır. Metodoloji, sınırlı veri probleminin üstesinden gelmek için yüksek güvenilirlikli örneklerin akıllı seçimi ve sürekli öğrenme stratejilerini birleştirmektedir. Bu yaklaşımın, hem hesaplama verimliliği hem de model performansı açısından önemli iyileştirmeler sağlaması beklenmektedir.

**Anahtar Kelimeler:** Transfer öğrenme, artımsal eğitim, yaş tahmini, otomatik etiketleme, OpenCLIP, catastrophic forgetting

---

## 1. GİRİŞ

Son yıllarda yapay zekâ ve makine öğrenmesi alanındaki hızlı gelişmeler, derin öğrenme modellerinin çeşitli uygulamalarda devrim niteliğinde ilerlemeler sağlamasına olanak tanımıştır. Özellikle bilgisayarla görü, doğal dil işleme ve konuşma tanıma gibi alanlarda elde edilen başarılar, yapay zekâ teknolojilerinin günlük yaşamın ayrılmaz bir bileşeni hâline gelmesinde kritik bir rol oynamaktadır [1,2]. Bununla birlikte, bu başarıların temelinde yatan büyük ölçekli derin öğrenme modellerinin geliştirilmesi ve uygulanması süreci, ciddi teknik ve operasyonel zorluklar içermektedir.

Bu zorlukların başında, büyük ölçekli derin öğrenme modellerinin sıfırdan eğitilmesinde karşılaşılan yüksek hesaplama gücü gereksinimi, uzun eğitim süreleri ve geniş veri kümesine olan ihtiyaç yer almaktadır. Modern derin öğrenme modelleri, milyonlarca hatta milyarlarca parametreye sahip olmakta ve bu parametrelerin optimal değerlerinin bulunması için yüksek düzeyde hesaplama kaynağı ve uzun süreli eğitim süreçleri gerektirmektedir. Bu durum, özellikle kaynak kısıtlı ortamlarda çalışan araştırmacılar ve geliştiriciler için önemli bir engel oluşturmaktadır [1].

Transfer öğrenme ve artımlı (incremental) eğitim yaklaşımları, söz konusu temel zorlukları aşmak üzere geliştirilen en etkili yöntemlerden birisidir. Transfer öğrenme, önceden eğitilmiş bir modelin edinmiş olduğu bilgi ve temsillerin yeni bir göreve uyarlanmasını mümkün kılarak, hem eğitim süresini önemli ölçüde kısaltmakta hem de daha az veri ile daha yüksek performans elde edilmesini sağlamaktadır [3,4]. Artımlı eğitim ise, bir modelin mevcut bilgisini koruyarak yeni verilerden öğrenmesini sağlayan teknikler olarak adlandırılabilir. Bu yaklaşım, modellerin değişen koşullara uyum sağlamasına imkân tanırken, daha önce edinilmiş bilgilerin unutulması (catastrophic forgetting) problemini büyük ölçüde engellemektedir [5,6].

Mevcut literatürde transfer öğrenme ve artımlı eğitim tekniklerinin bağımsız olarak uygulandığı pek çok başarılı çalışma mevcuttur. Ancak, bu iki yöntemin sistematik bir biçimde birleştirilerek gerçek zamanlı uygulamalarda birlikte kullanıldığı kapsamlı çalışmalar hâlen sınırlı sayıdadır. Özellikle kullanıcı geri bildirimlerinin artımlı eğitim sürecine entegrasyonu ve transfer öğrenme yaklaşımlarıyla birlikte optimize edilmesi, güncel araştırmalarda önemli bir boşluğu işaret etmektedir.

Bu çalışmanın temel motivasyonu, yaş tahmini gibi gerçek dünya uygulamalarında karşılaşılan pratik problemlere yenilikçi ve/veya alternatif bir çözüm sunmaktır. Yüz görüntülerinden yaş tahmini; güvenlik sistemlerinden sosyal medya uygulamalarına, pazarlama analizlerinden sağlık uygulamalarına kadar geniş bir yelpazede kullanılmaktadır. Ancak geleneksel yaklaşımlar, yüksek hesaplama maliyetleri, uzun geliştirme süreçleri ve sürekli değişen veri yapılarına adaptasyondaki zorluklar nedeniyle sınırlı bir etkinlik göstermektedir.

Görsellerden yaş tahmini konusu ile ilgili olarak InsightFace açık kaynak topluluğu tarafından geliştirilen Buffalo modeli, yüz tanıma görevlerinde en iyi seviye (state-of-the-art) performans gösteren güçlü bir derin öğrenme mimarisi olarak öne çıkmaktadır. Bu modelin yaş tahmini gibi farklı görevler için transfer edilmesi, hem akademik açıdan ilgi çekici bir problem hem de uygulamalı olarak yüksek potansiyele sahip bir çalışma alanı sunmaktadır.

Bu çalışma, hem akademik hem de endüstriyel açıdan bakıldığında transfer öğrenme ve artımsal eğitimin birleştirilmesi konusunda yeni bir metodoloji önermekte ve bu alandaki mevcut bilgi birikimine katkı sağlayacağı öngörülmektedir. Pratik açıdan ise, gerçek dünya uygulamalarında kullanılabilir, maliyet-etkin ve sürdürülebilir on premise AI çözümleri geliştirme konusunda değerli bir deneyim sunmaktadır. Yapay zeka model geliştirme süreçlerinde yeni bir bakış açısı sunarak aşağıdaki katkıları da sağlayacağı düşünülmektedir

Yenilikçi Hibrit Yaklaşım: Buffalo temel modeli çerçevesinde, transfer öğrenme ile artımsal eğitimin sistematik ve entegre bir şekilde uygulanması,
Akıllı Geri Bildirim Mekanizması: Kullanıcı geri bildirimlerine dayalı sürekli öğrenme süreçlerinin tasarımı ve etkin bir şekilde hayata geçirilmesi,
Gerçek Zamanlı Uygulama Altyapısı: Web platformu aracılığıyla model güncellemelerinin anlık olarak gerçekleştirilmesi ve dağıtımının sağlanması,
Kapsamlı Deneysel Analiz: UTKFace veri seti üzerinde detaylı performans değerlendirmeleri ve karşılaştırmalı deneylerin yürütülmesi,
Operasyonel ve Maliyet Verimliliği: Modelin performansını optimize ederken kaynak kullanımını minimize etmeye yönelik maliyet-etkin çözümlerin geliştirilmesi,
Çevresel Etkiler: Büyük ölçekli modellerin eğitim süresinin kısaltılmasıyla enerji tüketiminin azaltılması ve buna bağlı çevresel etkilerin en aza indirilmesi, 
Açık Kaynak Katkısı: Geliştirilen sistemin açık kaynak olarak paylaşılmasıyla akademik ve endüstriyel topluma katkı sağlanması. 

## 2. İLGİLİ ÇALIŞMALAR

Bu bölümde, önerilen yaklaşımın temel aldığı üç ana araştırma alanının mevcut durumu kapsamlı şekilde incelenmektedir: transfer öğrenme yaklaşımları, artımsal öğrenme teknikleri ve yaş tahmini uygulamaları. Her bir alan için mevcut literatür, öne çıkan yöntemler ve mevcut sınırlılıklar detaylı olarak analiz edilmektedir.

### 2.1. Transfer Öğrenme Yaklaşımları

Transfer öğrenme, makine öğrenmesi paradigması içerisinde, kaynak etki alanından (veya modelden) hedef etki alanına (veya modeline) bilgi aktarımını ifade eden bir süreçtir. Teorik temelleri 1990'lı yıllara dayansa da, derin öğrenme modellerinin yaygınlaşmasıyla birlikte pratik uygulamaları son on yılda hızlı ve belirgin bir artış göstermiştir.

Pan ve Yang [7] tarafından gerçekleştirilen kapsamlı sınıflandırmada, transfer öğrenme yöntemleri dört ana kategoriye ayrılmıştır: örnek (instance) transferi, özellik temsilinin (feature-representation) transferi, parametre transferi ve ilişkisel bilgi (relational knowledge) transferi. Bu sınıflandırma, transfer öğrenme alanındaki teorik çerçevenin temelini oluştururken, Weiss ve arkadaşları [8] farklı uygulama senaryolarını ayrıntılı biçimde incelemiş ve pratik rehberlik sağlamıştır.

Görüntü işleme alanında transfer öğrenmenin en başarılı uygulamalarından biri, ImageNet veri seti üzerinde önceden eğitilmiş konvolüsyonel sinir ağı (CNN) modellerinin çeşitli görevlerde kullanılmasıdır [9,10]. Özellikle, 2012 yılında ImageNet Challenge'da AlexNet'in elde ettiği başarı [10], derin öğrenme paradigmasının ve dolayısıyla transfer öğrenmenin pratik önemini ortaya koymuştur. Bu tarihten itibaren, ImageNet üzerinde eğitilmiş modeller, bilgisayar görü alanında pek çok görev için temel başlangıç noktası olarak kabul edilmiştir.

Yosinski ve arkadaşları [11] tarafından gerçekleştirilen öncü çalışma, transfer öğrenmenin etkinliğinin katman bazında nasıl değiştiğini deneysel olarak ortaya koymuş ve literatüre "transferability" kavramını kazandırmıştır. Bu çalışma, CNN'lerin alt katmanlarının genel ve evrensel özellikler öğrenirken, üst katmanların görev spesifik özellikler çıkardığı hipotezini güçlü deneysel kanıtlarla desteklemiştir.

Modern Transfer Öğrenme yaklaşımlarına göz atılacak olursa Fine-tuning Stratejileri; Önceden eğitilmiş modellerin farklı katmanlarının seçici olarak güncellenmesi, Multi-task Learning Stratejileri; Birden fazla görevi aynı anda öğrenen modeller, Domain Adaptation; Kaynak ve hedef domain arasındaki dağılım farklılıklarını ele alan yöntemler, Few-shot Learning; Çok az örnekle öğrenme kapasitesi olan modeller öne çıkmakta olup bu yaklaşımların her biri, farklı uygulama senaryolarında avantajlar sunmakta ancak aynı zamanda kendine özgü sınırlılıkları da bulunmaktadır.

### 2.2. Artımsal Öğrenme Teknikleri

Artımsal öğrenme (incremental learning), makine öğrenmesinde modellerin yeni verilerle sürekli güncellenmesini sağlayan teknikler bütünüdür. Bu alan, özellikle catastrophic forgetting probleminin çözümü etrafında şekillenmiş ve son yıllarda önemli teoretik ve pratik ilerlemeler kaydetmiştir.

Gepperth ve Hammer [12] tarafından yapılan kapsamlı survey, artımsal öğrenmenin temel prensiplerini tanımlarken, catastrophic forgetting probleminin neden kaynaklandığını ve mevcut çözüm yaklaşımlarını sistematik olarak incelemiştir. Bu çalışma, artımsal öğrenme alanının teorik temellerini oluştururken, gelecek araştırmalar için önemli bir yol haritası sunmuştur. 

Catastrophic Forgetting Problemi ve Çözüm Yaklaşımları konusu irdelenecek olursa birkaç farklı yaklaşım öne çıkmakta, Her bir yaklaşımın kendine özgü avantaj ve dezavantajları bulunmaktadır. Uygulama senaryosuna göre farklı yöntemler tercih edilmektedir. Yapay sinir ağlarının yeni görevler öğrenirken eski görevlerdeki performanslarını dramatik şekilde kaybetmesi çok sık rastlanan bir durumdur. Bu problem, artımsal öğrenmenin en kritik zorluğu olarak kabul edilmekte ve çözümü için çeşitli yaklaşımlar geliştirilmiştir:

Regularization-based Methods: Li ve Hoiem [13] tarafından önerilen Learning without Forgetting (LwF) yöntemi, yeni görevlerin öğrenilmesi sırasında önceki görevlerin çıktı dağılımlarının korunmasını hedeflemektedir. Bu yaklaşım, modelin eski görevlerdeki performansını korurken yeni bilgi edinmesini sağlar. Benzer şekilde, Kirkpatrick ve arkadaşları [14] tarafından geliştirilen Elastic Weight Consolidation (EWC) algoritması ise, Fisher Information Matrix kullanarak model parametrelerinin önem derecelerini belirler ve önemli parametrelerin büyük değişikliklerden korunmasını sağlayarak unutmayı minimize eder. Bu yöntem, öğrenme sürecinde modelin kritik bilgileri kaybetmeden yeni görevleri öğrenmesini mümkün kılar.

Rehearsal-based Methods: Bu yöntemler, önceki görevlerden elde edilen örneklerin saklanmasını veya jeneratif modeller aracılığıyla eski verilerin yeniden üretilmesini içerir. Saklanan örnekler veya yeniden oluşturulan veriler, Experience Replay mekanizmaları ile yeni öğrenme sürecine dahil edilerek modelin unutma eğilimi azaltılır. Bu sayede, model hem eski hem de yeni görevlerden gelen bilgileri birlikte kullanarak daha sağlam bir öğrenme gerçekleştirir.

Architecture-based Methods: Bu yöntemler, model mimarisinde değişiklik yaparak sürekli öğrenmeyi destekler. Örneğin, Progressive Neural Networks yapısı, her yeni görev için yeni alt ağlar ekleyerek bilgi aktarımını ve paylaşımını sağlar. Benzer şekilde, Dynamically Expandable Networks algoritması, modelin kapasitesini görevler arttıkça dinamik şekilde genişleterek öğrenme kapasitesini artırır ve unutmayı azaltır.

Meta-learning Approaches: Meta-öğrenme teknikleri, modelin farklı görevler arasında hızlı adaptasyon yeteneğini geliştirmeye odaklanır. Model-Agnostic Meta-Learning (MAML) yöntemi, model parametrelerini farklı görevlere hızlıca uyum sağlayacak şekilde optimize eder. Bunun yanı sıra, gradient tabanlı meta-öğrenme algoritmaları, öğrenme sürecinde görevler arası genelleme ve adaptasyon yeteneğini güçlendirir, böylece model yeni görevlere karşı daha esnek hale gelir.

### 2.3. Yaş Tahmini Uygulamaları ve Mevcut Yaklaşımlar

Yaş tahmini, bilgisayar görü alanında uzun süredir araştırılan ve çeşitli uygulamalara sahip önemli bir problem olarak karşımıza çıkmaktadır. Bu alandaki yöntemler, geleneksel makine öğrenmesi tekniklerinden başlayarak günümüzde yaygın şekilde kullanılan derin öğrenme tabanlı yaklaşımlara kadar geniş bir yelpazeyi kapsamaktadır.

Geleneksel Yaklaşımlar:
Erken dönem yaş tahmini çalışmaları, yüzün antropometrik özellikleri ve geometrik analizine dayanmaktadır. Bu yöntemlerde, yüz üzerindeki belirli anahtar noktalar arasındaki mesafeler, oranlar, çizgi ve açı ölçümleri gibi yapısal veriler kullanılarak yaş tahminleri yapılmıştır. Ancak, bu yaklaşımlar yüz ifadelerindeki varyasyonlar, farklı poz açıları ve aydınlatma koşullarındaki değişiklikler gibi çevresel faktörlere karşı oldukça hassas oldukları için sınırlı performans sergilemiştir. Bu da özellikle gerçek dünya koşullarında modelin genelleme kabiliyetini önemli ölçüde sınırlandırmıştır.

Modern Derin Öğrenme Yaklaşımları:
Son yıllarda yaş tahmini alanında derin öğrenme tekniklerinin yükselişi belirginleşmiştir. Bu bağlamda, Rothe ve arkadaşları [15] tarafından geliştirilen DEX (Deep EXpectation of apparent age) modeli, öncü çalışmalar arasında yer almaktadır. DEX, önceden eğitilmiş VGG-Face modelini yaş tahmini görevi için ince ayar (fine-tuning) sürecinden geçirerek yüksek başarı sağlamış ve yaş tahmininde derin öğrenmenin sunduğu üstün potansiyeli göstermiştir. Bu yaklaşım, karmaşık yüz özelliklerinin otomatik olarak öğrenilmesini mümkün kılarak, klasik yöntemlerin ötesinde bir genelleme kabiliyeti ortaya koymuştur.

Buffalo Modeli ve Özellikleri:
Buffalo modeli, InsightFace topluluğu tarafından geliştirilen ve yüz tanıma alanında yüksek performans gösteren state-of-the-art bir derin öğrenme mimarisidir [16]. Model, ResNet tabanlı bir encoder yapısına sahiptir ve 112x112 piksel boyutundaki RGB görüntüleri, 512 boyutlu ayırt edici (discriminative) özellik vektörlerine dönüştürme kapasitesine sahiptir.

Buffalo modelinin öne çıkan özellikleri şu şekilde sıralanabilir:
Yüksek Ayırt Edici Güç: Yüz tanıma görevlerinde son teknoloji performans sunması, modelin farklı bireyleri güvenilir şekilde ayırt etmesine olanak sağlar.
Dayanıklı Özellik Çıkarımı: Çeşitli yüz pozları, ifadeleri ve farklı aydınlatma koşullarına karşı sağlam ve tutarlı performans sergilemesi.
Transfer Learning Uygunluğu: Modelin önceden öğrenilmiş özelliklerinin, yaş tahmini gibi farklı görevler için adapte edilmesine uygun olması.
Verimli Mimari: Hesaplama kaynakları ve bellek kullanımı açısından optimize edilmiş yapısı, gerçek zamanlı uygulamalarda kullanımını kolaylaştırır.

Buffalo modelinin yaş tahmini problemine uyarlanması, transfer öğrenme paradigmasının pratikteki etkili bir örneğini teşkil etmekte; ayrıca, farklı veri dağılımlarına uyum sağlama (domain adaptation) süreçlerinde önemli kazanımlar sunmaktadır.

Bu esnek özelliklerinden dolayı uygulamada temel model olarak Buffalo seçilmiştir.

### 2.4. Mevcut Yaklaşımların Sınırlılıkları ve Araştırma Boşlukları

Literatür taraması neticesinde, yaş tahmini ve ilgili alanlarda kullanılan mevcut yöntemlerde çeşitli temel sınırlılıklar ve henüz yeterince ele alınmamış araştırma boşlukları belirlenmiştir. Bu kısıtlamalar hem yöntemsel hem de uygulamaya yönelik açılardan önem arz etmektedir.

Transfer Learning Sınırlılıkları:
Transfer öğrenme yaklaşımlarında en önemli problemlerden biri, kaynak (source) ve hedef (target) domainler arasındaki dağılım farklılıklarının (domain gap) üstesinden gelinmesidir. Bu farklılıklar, transfer edilen bilginin kalitesini doğrudan etkileyerek model performansını olumsuz yönde etkileyebilmektedir. Ayrıca, transfer sürecinde kullanılan ince ayar (fine-tuning) aşamalarında optimal öğrenme oranı (learning rate) belirleme ve dinamik olarak ayarlama problemleri ortaya çıkmakta; bu durum, öğrenme sürecinin verimliliğini ve kararlılığını sınırlandırmaktadır. Transfer kalitesinin objektif ve otomatik olarak değerlendirilmesi konusunda ise hâlihazırda yeterli yöntemler geliştirilmemiştir.

Incremental Learning Sınırlılıkları:
Artımsal öğrenme paradigmasında en kritik sorunlardan biri catastrophic forgetting olarak adlandırılan eski bilgilerin hızlıca unutulması problemidir ve bu durum halen tam anlamıyla çözülmüş değildir. Buna ek olarak, artımsal öğrenme yöntemlerinde bellek kullanımının etkin yönetilmesi (memory efficiency) ile hesaplama yükünün (computational overhead) dengelenmesi zorunlu olup, bu dengeyi sağlamak mevcut yöntemlerin çoğunda problem teşkil etmektedir. Uzun vadeli öğrenme süreçlerinde model performansının ve kararlılığının garanti altına alınmasına yönelik teorik ve pratik yaklaşımlar da yetersiz kalmaktadır.

Application-Specific Sınırlılıklar:
Gerçek dünya uygulamalarına entegre edilen modellerde, farklı ortam koşulları ve değişken kullanıcı davranışları karşısında dayanıklılık (robustness) eksikliği göze çarpmaktadır. Kullanıcı geri bildirimlerinin öğrenme döngüsüne etkin entegrasyonu için standartlaştırılmış prosedürler henüz gelişmemiştir, bu da sistemlerin adaptasyon yeteneğini kısıtlamaktadır. Ayrıca, ölçeklenebilirlik (scalability) ve gerçek zamanlı performans (real-time performance) optimizasyonu konusunda karşılaşılan zorluklar, uygulamaların pratik kullanıma geçişini engelleyen önemli zorluklardır.

Bu çalışma, literatürde belirlenen bu sınırlılıkları ve araştırma boşluklarını dikkate alarak, transfer öğrenme ile artımsal öğrenmenin sistematik ve bütüncül entegrasyonunu da hedefleyen kapsamlı bir çözüm sunmayı amaçlamaktadır. 

## 3. ÖNERİLEN METODOLOJİ

### 3.1. Sistem Mimarisi Genel Bakış

Önerilen metodoloji ile çeşitli kaynaklardan elde edilen veri kümelerinden hedef odaklı ve yüksek güvenilirliğe sahip örneklerin akıllı bir seçim sistemi aracılığıyla belirlenmesi amaçlanmaktadır. Bu seçilen kaliteli örneklerin, önerilen sistemde modelin geliştirilmesi sürecinde eğitim verisi olarak kullanılması önerilmekte ve böylece daha az veriyle modelin doğruluk oranının hızlı ve etkili bir şekilde artırılması hedeflenmektedir. Önerilen yaklaşım, yaş tahmini probleminde karşılaşılan temel zorluklardan biri olan sağlanabilir sınırlı kaliteli veri sorununun üstesinden gelmek için yenilikçi bir otomatik etiketleme yaklaşımı benimsemektedir.

### 3.2. Veri İşleme ve Ön Hazırlık

Yaş tahmini problemine özgü olarak tasarlanan önerilen metodolojide, sistemin öncelikli olarak farklı türdeki görsel verileri (resim ve video formatları) ayırt edebilme yeteneği kazandırılması gerekmektedir. Bu ayrıştırma işleminin, dosya türü denetimi (MIME Type Audit) ile gerçekleştirilebileceği önerilmektedir. 

Video dosyalarının işlenmesi sırasında ise belirli zaman aralıklarında kareler çıkarılarak her bir karenin ayrı bir görüntü olarak ele alınması önerilmektedir. Bu süreçte, görüntü işleme kütüphaneleri kullanılarak saniyede belirli sayıda kare alınması (örnekleme metodu ile işlem hızı arttırımı) sağlanabilmesi hedeflenmektedir. Resim dosyalarının aynı analiz süreçlerinden geçirileceği için, bundan sonraki aşamalarda yöntemin video verisi üzerinden açıklanması önerilmektedir.

### 3.3. Yüz Tespiti ve Özellik Çıkarma

Elde edilen görüntülerin, öncelikle temel analizden geçirilmesi gerekmektedir. Bu aşamada gelişmiş yüz tanıma teknolojisi (InsightFace) kullanılarak yüz tespiti yapılması önerilmektedir. Önerilen yaklaşımda, sistemin ayarlanabilir hassasiyet parametreleri ile farklı koşullarda optimal performans gösterebilecek şekilde tasarlanması önerilmektedir. 

Tespit edilen her yüz için konum bilgileri (bounding box), özellik vektörleri (embedding), yüz anatomik noktaları (landmark) ve başlangıç yaş tahmininin çıkarılması gerekmektedir. Bu özellikler, hem anlık analiz hem de gelecekteki model geliştirme süreçleri için kritik öneme sahip olmaktadır.

### 3.4. İkili Model Yaş Tahmini ve Güvenilirlik Değerlendirmesi

Önerilen metodolojide, sistemin iki farklı yaş tahmini modelini birlikte kullanacak şekilde tasarlanması önerilmektedir. İlk model olarak, geniş veri kümeleri üzerinde önceden eğitilmiş bir yaş tahmini sistemi (Buffalo_l) kullanılması önerilmektedir. Bu model genel durumlarda başarılı olmasına rağmen, özel kullanım durumlarında sınırlı performans gösterebileceği değerlendirilmektedir. 

Bu eksikliği gidermek amacıyla, özel olarak geliştirilmiş ikinci bir yaş tahmini modelinin (Custom Age Head) devreye girmesi önerilmektedir. Bu modelin, ilk modelden elde edilen özellik vektörlerini kullanarak daha hassas yaş tahminleri üretebilmesi hedeflenmektedir.

Her yaş tahmini için güvenilirlik değerlendirmesinin, gelişmiş bir dil-görsel anlayış sistemi (OpenCLIP) kullanılarak yapılması önerilmektedir. Bu yöntemde, spesifik yaş tahminleri için aşağıdaki prompt yapısının kullanılması önerilmektedir:

**Hedef Prompt:** "this face is [predicted_age] years old"
**Karşıt Promptlar:** Farklı yaş gruplarından seçilen alternatif yaş tahminleri
- Genç tahmin için: "this face is 45 years old", "this face is 65 years old"
- Orta yaş tahmin için: "this face is 18 years old", "this face is 70 years old"

Bu sürecin, modelin tahmini ile görsel içerik arasındaki anlamsal uyumluluğu objektif olarak değerlendirebilmesi beklenmektedir.

### 3.5. Otomatik Etiketleme ve Kaliteli Veri Seçimi

Önerilen sistemin en özgün özelliği, otomatik etiketleme (pseudo-labeling) yaklaşımı ile kaliteli eğitim verisi oluşturma metodolojisinin kullanılması gerektiğidir. İlk modelin yaş tahminleri için hesaplanan güvenilirlik skorlarının, önceden belirlenen bir eşik değeri ile karşılaştırılması önerilmektedir. Yüksek güvenilirlik skoruna sahip tahminlerin, otomatik etiket olarak işaretlenmesi ve ikinci modelin geliştirilmesi sürecinde eğitim verisi olarak kullanılması önerilmektedir.

Bu yaklaşımın, geleneksel eğitim yöntemlerine kıyasla önemli avantajlar sağlayabileceği değerlendirilmektedir. Manuel etiketleme gereksinimini büyük ölçüde azaltabilmesi ve büyük veri kümelerinden otomatik olarak kaliteli örnekler seçebilmesi beklenmektedir. Ayrıca, modelin sürekli öğrenme kapasitesinin artırılarak kendini geliştirmesinin mümkün kılınması hedeflenmektedir. Önerilen sistemin ayrıca farklı veri dağılımlarına uyum sağlama yeteneği geliştirebileceği öngörülmektedir.

### 3.6. Sürekli Öğrenme ve Bilgi Kaybı Önleme

Önerilen metodolojide, otomatik etiketleme yaklaşımının sürekli öğrenme (incremental learning) prensiplerine dayanması önerilmektedir. İkinci modelin (Custom Age Head) yeni otomatik etiketli veriler ile geliştirilmesi sürecinde, önceden öğrenilen bilgilerin kaybolması (catastrophic forgetting) sorununun ele alınması kritik önem taşımaktadır.

**Bu sorunun çözümü için aşağıdaki stratejilerin uygulanması önerilmektedir:**

**Base Model Koruma Yaklaşımı:** İlk modelin (Buffalo) parametrelerinin dondurulması (freezing) önerilmekte, bu sayede önceden öğrenilmiş genel yüz özellik çıkarımı yeteneklerinin korunması hedeflenmektedir. Bu yaklaşımın, temel özellik çıkarım yeteneklerini korurken hesaplama verimliliği sağlayacağı öngörülmektedir.

**Seçici İnce Ayar Stratejisi:** Sadece özelleşmiş yaş tahmini katmanlarının güncellenmesi önerilmekte, bu yaklaşımın hem hesaplama verimliliği sağlayacağı hem de temel özellik çıkarım yeteneklerini koruyacağı öngörülmektedir. Bu strateji, Elastic Weight Consolidation (EWC) prensiplerini takip ederek kritik parametrelerin korunmasını garanti edecek şekilde tasarlanması önerilmektedir.

**Deneyim Tekrarı Mekanizması:** Her eğitim döngüsünde orijinal eğitim verilerinin belirli bir oranının (%20-30) korunarak yeni pseudo-label verilerle birlikte kullanılması önerilmektedir. Ayrıca, ağırlıklı öğrenme yaklaşımı ile yüksek güvenilirlik skoruna sahip örneklere daha fazla ağırlık verilmesi gerektiği değerlendirilmektedir.

**Regularization ve Kararlılık Kontrolleri:** Model parametrelerinde ani değişikliklerin önlenmesi için kontrollü öğrenme oranlarının (adaptive learning rates) kullanılması önerilmektedir. Sistemin, aşırı öğrenme riskini minimize eden düzenleme tekniklerinin (L2 regularization ve dropout) uygulaması gerektiği değerlendirilmektedir. Early stopping mekanizması ile optimal eğitim süresinin belirlenmesinin, modelin genelleme yeteneğinin korunmasını sağlayacağı öngörülmektedir.

### 3.7. Karşılaştırmalı Değerlendirme ve Model Seçimi

Önerilen sistemde, iki farklı yaş tahmini modelinin sonuçlarının karşılaştırmalı değerlendirme (çapraz doğrulama) yöntemi ile analiz edilmesi önerilmektedir. Bu süreçte her modelin kendi tahmini için güvenilirlik skoru hesaplanması, ardından her modelin diğer modelin tahmini için de güvenilirlik değerlendirmesi yapılması gerekmektedir. 

**Çapraz Doğrulama Mekanizması:** Buffalo modelinin tahmin ettiği yaş için Custom modelin güven skorunun hesaplanması ve vice versa işleminin gerçekleştirilmesi önerilmektedir. Net güven skorunun (kendi güveni - karşıt güveni) hesaplanarak en yüksek net skora sahip modelin seçilmesi gerektiği değerlendirilmektedir.

Bu birleştirici yaklaşımın (ensemble), model performansını objektif olarak değerlendirebilmesi ve en güvenilir sonucu seçebilmesi beklenmektedir. Karşılaştırmalı değerlendirme sürecinin, aynı zamanda sistemin otomatik etiketleme kalitesini kontrol etmek için de kullanılabileceği öngörülmektedir. Tutarsız tahminlerin tespit edildiği durumların belirlenmesi ve bu örneklerin eğitim veri setinden çıkarılması gerektiği değerlendirilmektedir.

Bu metodoloji, yaş tahmini problemi için hem kuramsal hem de uygulamalı açıdan yenilikçi bir çözüm sunabileceği değerlendirilmektedir. Otomatik etiketleme ile kaliteli veri seçimi, sürekli öğrenme ile devamlı iyileşme ve bilgi kaybı önleme stratejilerinin bir araya gelerek güçlü ve uyarlanabilir bir sistem oluşturabileceği öngörülmektedir. 

## 4. SONUÇ VE DEĞERLENDİRME

Önerilen metodoloji, transfer öğrenme ve artımsal eğitim tekniklerinin sistematik birleştirilmesi yoluyla yaş tahmini problemine yenilikçi bir çözüm sunmaktadır. Bu yaklaşımın temel katkıları şu şekilde özetlenebilir:

**Metodolojik Katkılar:**
- Buffalo temel modeli ile Custom Age Head'in hibrit kullanımı
- OpenCLIP tabanlı güven skorlaması ile otomatik etiketleme mekanizması
- Çapraz doğrulama ile model seçimi stratejisi
- Catastrophic forgetting'e karşı çok katmanlı koruma yaklaşımı

**Beklenen Avantajlar:**
Önerilen metodolojinin uygulanması durumunda, geleneksel yaklaşımlara kıyasla önemli iyileştirmeler sağlanması beklenmektedir. Manuel etiketleme ihtiyacının %60-80 oranında azaltılabileceği, model eğitim süresinin %70-85 oranında kısaltılabileceği ve kaynak kullanımının optimize edilebileceği öngörülmektedir.

**Gelecek Çalışmalar:**
Bu metodolojinin deneysel doğrulaması, farklı veri setleri üzerinde kapsamlı testlerin gerçekleştirilmesi ve gerçek dünya uygulamalarında performansının değerlendirilmesi gelecek çalışmaların odak noktasını oluşturmaktadır.

---

## KAYNAKÇA

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.

[2] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.

[3] Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

[4] Zhuang, F., Qi, Z., Duan, K., Xi, D., Zhu, Y., Zhu, H., ... & He, Q. (2020). A comprehensive survey on transfer learning. Proceedings of the IEEE, 109(1), 43-76.

[5] Chen, Z., & Liu, B. (2018). Lifelong machine learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 12(3), 1-207.

[6] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. Neural Networks, 113, 54-71.

[7] Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

[8] Weiss, K., Khoshgoftaar, T. M., & Wang, D. (2016). A survey of transfer learning. Journal of Big Data, 3(1), 1-40.

[9] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009, June). Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 248-255).

[10] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6), 84-90.

[11] Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks?. Advances in Neural Information Processing Systems, 27.

[12] Gepperth, A., & Hammer, B. (2016, September). Incremental learning algorithms and applications. In European Symposium on Artificial Neural Networks (ESANN).

[13] Li, Z., & Hoiem, D. (2017). Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12), 2935-2947.

[14] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521-3526.

[15] Rothe, R., Timofte, R., & Van Gool, L. (2018). Deep expectation of real and apparent age from a single image without facial landmarks. International Journal of Computer Vision, 126(2-4), 144-157.

[16] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4690-4699). 