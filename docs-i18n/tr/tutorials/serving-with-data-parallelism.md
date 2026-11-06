---
translation_status: translated
---

# Veri paralelliğiyle model sunma { #serving-models-with-data-parallelism }

Bu öğretici, vLLM tariflerindeki veya SGLang cookbook içindeki veri paralelliği yapılandırmalarını GPUStack'te kullanmayı açıklar. GPU seçimi, arka uç parametreleri, iş yüklerini inceleme ve kurtarma adımlarını kapsar.

## Ön koşullar { #prerequisites }

- GPUStack kurulu ve çalışır durumda olmalıdır. Sunucuyu, katılan worker'ları ve UI'ı bu iş yüklerini destekleyen sürümlere güncelleyin. Sürüm uyuşmazlığı bildirimleri, özellik uyumluluğunu kontrol etmenin yerini tutmaz.
- **Her birinde aynı modelden 4 NVIDIA GPU ve GPU başına 80 GB VRAM bulunan 2 Linux worker** hazırlayın. Normal vLLM dağıtımı sekiz GPU'nun tamamını, normal SGLang dağıtımı bir worker'ı kullanır. PD örneğinde bir worker Prefill'e, diğeri Decode'a ayrılır.
- Cihazlarınıza uygun bir GPUStack Runner imajı kullanarak `Backend Version` için vLLM'de `0.32.0`, SGLang'de `0.5.22` seçin. Normal SGLang dağıtımının eşlik eden SMG'si, `sglang-router=0.3.2` içeren imaj bağımlılık meta verisini de gerektirir. Paketi özel bir imaja yüklemek tek başına bu kontrolü karşılamaz.
- Worker'lar Hugging Face'ten model indirebilmelidir. Yerel modeller için katılan tüm worker'larda aynı model yolunu kullanın.
- Worker'lar kayıtlı IP adresleri üzerinden haberleşebilmeli ve ağ geçidi, çıkarım API portlarına erişebilmelidir. Worker'lar arasında yüksek hızlı bağlantı önerilir. PD için [ağ gereksinimlerini](../user-guide/pd-disaggregation.md#networking-requirements) de karşılayın.

vLLM ve SGLang DPA/EP örnekleri `Qwen/Qwen3-30B-A3B`, normal SGLang DP örneği ise `Qwen/Qwen3-8B` kullanır. Bu örnekler kaynak yerleşimini gösterir. Kullanılabilir bağlam uzunluğunu ve eşzamanlılığı kendi ortamınızda doğrulayın; ikisi de ağırlık hassasiyetine, KV önbelleğine, iletişim kitaplıklarına ve cihaz özelliklerine bağlıdır.

## Adım 1: Modeli dağıtın { #step-1-deploy-the-model }

Aşağıdaki örnekler her dağıtımın GPUStack'te nasıl yapılandırılacağını gösterir. Başka bir örneği denemeden önce önceki dağıtımı durdurup GPU'larını serbest bırakın.

### vLLM External-LB ile dağıtın { #deploy-with-vllm-external-lb }

1. `Deployments` sayfasında `Deploy Model` düğmesine tıklayın, `Hugging Face` seçin ve `Qwen/Qwen3-30B-A3B` arayın.
2. `Backend` değerini `vLLM`, `Backend Version` değerini `0.32.0` olarak ayarlayın.
3. `Allow Distributed Inference Across Workers` seçeneğini etkinleştirin. `Scheduling` altında `Scheduling Mode` değerini `Specify GPU` yapın. `GPU Selector` ile her worker'dan dört GPU seçin.
4. `Backend Parameters` listesine şu girdileri ekleyin:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

5. `Save` düğmesine tıklayın.

Bu yerleşim `DP × TP = 4 × 2 = 8` GPU kullanır ve her worker'da iki tam rank bulunur. GPUStack cihazları ve portları atar, her rank için başlatma argümanlarını oluşturur ve örneğin tamamı hazır olduğunda dört rank API'sini yayımlar.

### vLLM Multi-port External-LB ile dağıtın { #deploy-with-vllm-multi-port-external-lb }

1. External-LB örneğindeki modeli, sürümü, GPU seçimini ve diğer parametreleri koruyun.
2. `--data-parallel-external-lb` yerine şunu kullanın:

    ```bash
    --data-parallel-multi-port-external-lb
    ```

3. `Save` düğmesine tıklayın.

Dağıtım yine dört bağımsız rank API'si sunar. Her worker'ın yerel DPSupervisor'ı iki yerel rank'ı yönetir. Toplu sağlık portu, paylaşılan bir çıkarım API'si değildir.

Her rank bütünüyle tek worker'a sığmalıdır. Tüm worker'ların local DP değeri aynı ve en az `2` olmalıdır. Açıkça ayarlanırsa `--api-server-count` ve `--prefill-context-parallel-size` değerleri `1` olmalıdır. Bu mod `--headless`, `--grpc` veya `--uds` desteklemez.

### vLLM Hybrid-LB ile dağıtın { #deploy-with-vllm-hybrid-lb }

1. External-LB örneğindeki modeli, sürümü ve GPU seçimini koruyun.
2. `Backend Parameters` listesinin tamamını şununla değiştirin:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-hybrid-lb
    --max-model-len=8192
    ```

3. `Save` düğmesine tıklayın.

Bu örnek, her biri iki yerel rank'ı yöneten iki paylaşılan API sunar. Ağ geçidi paylaşılan API'yi, vLLM de onun arkasındaki rank'ı seçer. Paylaşılan API'nin istek sayısı, tek bir rank'ın istek sayısı değildir.

Normal External-LB ve Hybrid-LB, worker'lar arasında TP/PP parçalarını da destekler. Bir rank birden fazla worker'a yayıldığında her düğüme eşit sayıda cihaz atanmalıdır. Bazı worker'lar bu durumda yalnızca hesaplama yapar, API sunmaz. Gerçek iş yükü yerleşimini kontrol edin.

### SGLang DP ve SMG ile dağıtın { #deploy-with-sglang-dp-and-smg }

1. `Deployments` sayfasında `Deploy Model` düğmesine tıklayın, `Hugging Face` seçin ve `Qwen/Qwen3-8B` arayın.
2. `Backend` değerini `SGLang`, `Backend Version` değerini `0.5.22` olarak ayarlayın.
3. `Scheduling` altında `Scheduling Mode` değerini `Specify GPU` yapın. `GPU Selector` ile aynı worker'dan dört GPU seçin.
4. `Backend Parameters` listesine şu girdileri ekleyin:

    ```bash
    --tp-size=1
    --dp-size=4
    --context-length=8192
    --policy=cache_aware
    ```

5. `Save` düğmesine tıklayın.

Normal DP, dört tam model kopyasını `DP × TP = 4 × 1 = 4` GPU üzerinde çalıştırır. GPUStack bir Engine ve yalnızca CPU kullanan bir SMG bileşeni oluşturur. Ağ geçidi SMG'ye bağlanır. Ayrı bir SMG dağıtımı veya elle `--worker-urls` ayarı gerekmez. Bu örnek DP attention etkinleştirmediği için dört GPU da aynı worker'da olmalıdır.

`--policy` yalnızca eşlik eden SMG'ye uygulanır, Engine'e aktarılmaz. Varsayılan değer `cache_aware` olup `random`, `round_robin`, `power_of_two` ve `manual` da desteklenir. Aynı öneke sahip istekler sürekli aynı hedefe gidebilir. İsteklerin eşit dağılmaması tek başına yönlendirme arızası anlamına gelmez.

### SGLang DPA ve EP ile dağıtın { #deploy-with-sglang-dpa-and-ep }

1. Önceki örnekteki tek worker üzerindeki dört GPU seçimini ve SGLang sürümünü koruyun. Modeli `Qwen/Qwen3-30B-A3B` olarak değiştirin.
2. `Backend Parameters` listesinin tamamını şununla değiştirin:

    ```bash
    --tp-size=4
    --dp-size=1
    --attn-dp-size=4
    --ep-size=4
    --context-length=8192
    --policy=cache_aware
    ```

3. `Save` düğmesine tıklayın.

Bu örnek, dört GPU üzerinde attention DP ile EP'yi birleştirir. Normal DP formülü `DP × TP` ile on altı GPU hesaplamayın. `--attn-dp-size` ve `--attention-data-parallel-size` eş adlardır; bilinen Runner sürümleri için SGLang `0.5.22` veya sonrası gerekir. Eski sürümler `--dp-size=4 --enable-dp-attention` kullanır. Aynı paralelliği iki kez tanımlamak için iki sözdizimini birleştirmeyin.

Normal bir SGLang dağıtımında etkin yönlendirme rank sayısı, DP size ile attention DP size çarpımıdır. Yeni DPA sözdiziminde TP size, attention DP size değerine tam bölünmelidir ve iki DP değeri de aynı anda `1` değerini aşamaz; bkz. [SGLang Activation Logic](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide#activation-logic). GPUStack yalnızca yönlendirme rank sayısı `1` değerini aştığında SMG oluşturur. Yönlendirme rank sayısı `1` olan yalnızca EP yapılandırması doğrudan Engine'i kullanır ve normal SMG `--policy` parametresini ayarlayamaz. SGLang PD ayrı bir PD router kullanır; her role normal dağıtımın SMG'sini eklemez.

### Topolojiyi GPUStack'in yönetmesine izin verin { #let-gpustack-manage-the-topology }

Bu yönetilen örnekler, örneğin tamamı için paralelliği, modu ve GPU seçimini yapılandırır. Rank'ları, düğüm numaralarını, iletişim adreslerini ve portları GPUStack oluşturur. Her worker için ayrı dağıtım oluşturmak veya `GPUSTACK_SKIP_GPU_COUNT_CHECK` ayarlamak gerekmez.

`--data-parallel-rank`, `--data-parallel-start-rank`, `--data-parallel-address`, `--data-parallel-rpc-port`, `--nnodes`, `--node-rank`, `--host` ve `--port` ayarlarını GPUStack'e bırakın. `--data-parallel-size-local` açıkça ayarlanabilir, ancak gerçek kaynak atamasıyla eşleşmelidir.

!!! warning

    Üç vLLM LB bayrağı birbirini dışlar. Yönetilen rank API'leri TCP HTTP kullanır ve Engine tarafında `--ssl-keyfile` veya `--ssl-certfile` desteklemez; istemci TLS'ini ağ geçidinde yapılandırın. `GPUSTACK_MANUAL_DP=1` yalnızca normal yerleşik vLLM arka ucuyla elle DP içindir, PD veya SGLang için değildir. Değiştirilmesi, örneklerin yeniden oluşturulmasını gerektirir. Elle uygulanan akış için [MultiProcessing öğreticisine](running-distributed-vllm-with-multiprocessing.md#step-2-deploy-the-model-with-the-mp-backend) bakın.

## Adım 2: İş yüklerini ve hazır olma durumunu kontrol edin { #step-2-check-workloads-and-readiness }

1. `Deployments` içinde dağıtımı genişletin ve örneğin durumunu kontrol edin.
2. Örnek bilgilerinde `Workloads / ranks` bölümünü açın. Worker'ı, cihazları, rank/component bilgisini ve API'nin hazır olma durumunu kontrol edin.
3. `Running` durumunu bekleyin. İndirmenin tamamlanması, konteynerin başlaması ve API'nin hazır olması ayrı aşamalardır.
4. Arıza olursa worker'ı ve nedeni belirlemek için iş yükünün günlük girişini veya örneğin `View Logs` seçeneğini kullanın. Günlükleri dışa aktarmak için `Download Logs` seçin.

| Örnek | Beklenen yerleşim |
| --- | --- |
| vLLM External-LB / Multi-port External-LB | Dört rank API'si |
| vLLM Hybrid-LB | Her biri iki yerel rank'ı yöneten iki paylaşılan API |
| SGLang DP | Engine ve SMG bileşenleri; Engine içinde dört mantıksal rank |
| SGLang DPA/EP | Engine ve SMG bileşenleri; bu örnekte dört mantıksal rank |

Günlükleri ve metrikleri gerçek kaynaklarına göre okuyun. SGLang'in mantıksal DP görünümleri, paylaşılan Engine'in yerel `dp_rank` metriklerini filtreler. Bu boyutu içermeyen metrikler veri göstermeyebilir. Hybrid'in paylaşılan API metriklerini veya Engine'in toplu metriklerini rank sayısıyla çarpmayın. Eksik metrikler veya geçici olarak sıfır istek sayısı tek başına Engine arızası anlamına gelmez.

## Adım 3: Playground'da doğrulayın { #step-3-verify-in-playground }

1. Hazır dağıtım için `Open in Playground` seçeneğine tıklayın.
2. Dağıtımın modelinin seçili olduğunu doğrulayın. “Veri paralelliğini tek cümleyle açıkla.” yazıp gönderin.
3. Yanıt geldiğini doğrulayın, ardından istek yolunu incelemek için iş yükü günlüklerini ve izlemeyi kontrol edin.

## PD ayrıştırmasıyla kullanma { #use-with-pd-disaggregation }

PD, Prefill ve Decode'u ayrı yapılandırır ve kaynaklarını ayrı ayırır. Bir PD router, istekleri ve KV aktarımını koordine eder. Normal dağıtımda ağ geçidinden rank'a doğrudan yönlendirme, PD hizmetinin tamamına uygulanmaz: PD'nin giriş noktası her zaman router'dır.

### Aktarım yöntemini ve yerleşimi kontrol edin { #check-the-transport-and-layout }

| vLLM modu | Router üyeleri | Aktarım kısıtları |
| --- | --- | --- |
| External-LB | Her DP rank'ının ana API'si | CUDA NIXL, CUDA Mooncake, Ascend Mooncake; CUDA Mooncake her rank'ın tek worker'a sığmasını gerektirir |
| Multi-port External-LB | Yerel DPSupervisor'ın başlattığı rank ana API'leri; her biri DP=1 olarak kaydedilir | Yalnızca CUDA NIXL; tüm worker'larda eşit local DP ≥ 2 ve her rank tamamen tek worker'da olmalıdır |
| Hybrid-LB | API sunan worker'lardaki paylaşılan uç noktalar; her biri kendi local DP değeriyle kaydedilir | CUDA NIXL, CUDA Mooncake, Ascend Mooncake; CUDA Mooncake'in statik router'ı tüm paylaşılan API'lerde aynı local DP değerini gerektirir |

### Bir vLLM NIXL grubu dağıtın { #deploy-a-vllm-nixl-group }

1. `Qwen/Qwen3-30B-A3B` için dağıtım oluşturun. `vLLM` ve `Backend Version` `0.32.0` seçin, `PD Disaggregation` etkinleştirin ve `Transport` altında `NIXL` seçin.
2. `Roles` altında hem `Prefill` hem `Decode` için `Parameters and environment` değerini `Custom` yapın. Her rolün `Backend Parameters` alanına aşağıdaki tam listeyi girin. Model düzeyindeki listeyi boş bırakın ve platformun yönettiği `Router` bileşenini koruyun:

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=2
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

3. Her rolün `Resources and scheduling` değerini `Custom` yapın. `Specify GPU` ile ilk worker'daki dört GPU'yu Prefill'e, ikinci worker'daki dört GPU'yu Decode'a atayın.
4. `Save` düğmesine tıklayın. Prefill, Decode ve Router hazır olduğunda Adım 3'teki gibi Playground'da doğrulayın.

Prefill ve Decode'un her birinde iki rank bulunur; toplam sekiz GPU kullanılır. Rolün parametre listesi model düzeyindeki listeye eklenmez, listenin tamamının yerini alır. Seçilen tarif KV bağlayıcısını, rol adreslerini ve bağlantı parametrelerini oluşturur. Eş adreslerini elle girmek gerekmez.

Bu CUDA NIXL örneğinde, her iki roldeki `--data-parallel-external-lb` yerine `--data-parallel-multi-port-external-lb` kullanarak her tarafta iki bağımsız rank API'sini koruyun. Alternatif olarak `--data-parallel-hybrid-lb` kullanıp her role `--data-parallel-size-local=2` ekleyin. Her taraf bu durumda iki rank'ı yöneten tek bir paylaşılan API sunar. Aynı anda yalnızca bir LB bayrağı kullanın.

Her iki tarafta kullanılabilir üyeler oluşunca Router başlar. Başlangıçta `Partially ready` normaldir. Tek rank veya rolün hazır olması, PD isteğinin tüm yolunun çalıştığını kanıtlamaz. CUDA Mooncake statik üyelik kullanır; üye değişiklikleri router'ın yeniden oluşturulmasına ve isteklerin kısa süre kesilmesine neden olabilir. Rol yönetimi ve KV aktarım yapılandırması için [PD ayrıştırmasına](../user-guide/pd-disaggregation.md) bakın. SGLang PD yönlendirmesini, normal SMG örneğinin Engine `--policy` parametresiyle değil, Router'ın `Route arguments` alanıyla yapılandırın.

## Adım 4: Dağıtımı güncelleyin ve kurtarın { #step-4-update-and-recover-the-deployment }

### Yapılandırma değişikliğini uygulayın { #apply-a-configuration-change }

Paralellik, LB modu veya GPU yerleşimi değiştiğinde örneğin tamamı yeniden oluşturulmalıdır.

1. Yeni yapılandırmayı kaydedin.
2. Dağıtımın `Operations` menüsünden `Stop` seçin.
3. Eski örneklerin kaldırılmasını bekleyin. PD durdurma akışı, çevrimiçi worker'ların eski iş yüklerinin durduğunu doğrulamasını bekler.
4. `Start` seçin, yeni örnekler hazır olunca Playground'da doğrulayın. Zamanlanmış ölçeklendirmesi etkin normal bir dağıtımda, geçerli zamanlamayı sürdürmek için `Resume Schedule` kullanın. Hedef sayı hâlâ sıfır olabilir.

| Durum | Kurtarma davranışı |
| --- | --- |
| Normal yönetilen örnekte Engine/rank arızası | Örneğin tüm uç noktaları kaldırılır. Eski iş yüklerinin durduğu doğrulandıktan sonra `Auto-Restart On Error` ayarına göre örneğin tamamı kurtarılır. Normal örnekler, ilk worker/GPU ataması üzerinde güncel yapılandırmayla kurtarılabilir. |
| Normal SGLang dağıtımında yalnızca SMG arızası | Sağlıklı Engine korunarak yalnızca SMG yeniden oluşturulur. |
| Yapılandırma değişmeden PD Engine örneğinin arızalanması | Etkilenen örneğin tamamı kurtarılır, sağlıklı karşı rol ve diğer örnekler korunur, router üyeliği güncellenir. |
| PD yapılandırmasının değişmesi | Eski yapılandırmadaki üyelerin otomatik kurtarılması ve yeniden oluşturulması duraklatılır. Kaydedilen yapılandırmayı uygulamak için tüm dağıtımı durdurup başlatın. |
| Worker'a ulaşılamaması ve eski iş yükü durumlarının bilinmemesi | Eski iş yüklerinin sahipliği korunur, durdurma onayı beklenir. Başka worker'a otomatik taşıma garanti edilmez. |

Değişiklikleri kaydetmek sağlıklı süreçleri hemen yeniden başlatmaz. Normal arıza kurtarması yeni GPU seçmez. Yeni yapılandırma farklı kaynak yerleşimi gerektiriyorsa kurtarmanın yeniden zamanlamasına güvenmek yerine dağıtımı açıkça yeniden oluşturun.

### Kullanılamayan worker'ı değiştirin { #replace-an-unavailable-worker }

Ulaşılamayan worker, GPUStack'in o anda düğümün veya iş yüklerinin durumunu doğrulayamadığı anlamına gelir. Bu durum, düğümün onarılıp onarılamayacağını göstermez. Düğüm veya ağ incelemesi worker'ın değiştirilmesi gerektiğini ortaya koyduğunda:

1. Dağıtım için `Stop` veya `Delete` seçin. GPUStack ulaşılamaz, heartbeat süresi dolmuş veya kaldırılmış olarak değerlendirilen worker'ların eski örnek kayıtlarını siler ve çevrimiçi worker'ların eski iş yüklerini durdurmasını bekler.
2. Dağıtımı koruyorsanız yeni bir worker kaydedin ve `GPU Selector` içindeki GPU seçimini güncelleyin.
3. `Start` seçmeden önce eski örneklerin kaldırılmasını bekleyin. Dağıtımı sildiyseniz yeniden oluşturun.

!!! warning

    Ulaşılamayan worker'ın örnek kayıtlarını silmek, uzak süreçlerinin durduğu anlamına gelmez. Eski makine çalışıyor ancak ağdan yalıtılmışsa, yeni grubu başlatmadan önce eski iş yüklerini makine veya altyapı düzeyinde durdurun ya da yalıtın. Eski worker yeniden bağlandığında geçerli örneklere artık ait olmayan eski iş yüklerini temizler. Kurtarma, KV önbelleğinin veya devam eden isteklerin korunmasını garanti etmez.

## Kaynaklar { #references }

- [vLLM 0.32.0: Data Parallel Deployment](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/)
- [SGLang: DP, DPA and SGLang DP Router](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
