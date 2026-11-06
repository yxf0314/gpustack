---
translation_status: translated
---

# データ並列によるモデルの提供 { #serving-models-with-data-parallelism }

このチュートリアルでは、vLLM のレシピや SGLang cookbook のデータ並列構成を GPUStack で使用する方法を説明します。GPU の選択、バックエンドパラメーター、ワークロードの確認、障害復旧を扱います。

## 前提条件 { #prerequisites }

- GPUStack が稼働していること。サーバー、参加する Worker、UI を、このワークロードに対応したバージョンに更新してください。バージョン不一致の通知だけでは、機能の互換性を確認できません。
- **Linux Worker を 2 台用意し、それぞれに同一モデルの NVIDIA GPU を 4 基、各 GPU に 80 GB の VRAM を搭載してください**。通常の vLLM デプロイでは 8 基すべてを使用します。通常の SGLang デプロイでは 1 台の Worker を使用します。PD の例では、Prefill と Decode に 1 台ずつ割り当てます。
- デバイスに対応する GPUStack Runner イメージを使用し、`Backend Version` を vLLM では `0.32.0`、SGLang では `0.5.22` に設定してください。通常の SGLang デプロイに付属する SMG には、`sglang-router=0.3.2` を含むイメージ依存関係メタデータも必要です。カスタムイメージにパッケージをインストールするだけでは、このチェックを満たしません。
- Worker から Hugging Face のモデルをダウンロードできること。ローカルモデルでは、すべての参加 Worker で同じモデルパスを使用してください。
- 登録済み IP アドレスを通じて Worker 間で通信でき、ゲートウェイから推論 API ポートにアクセスできること。Worker 間には高速な接続を推奨します。PD では[ネットワーク要件](../user-guide/pd-disaggregation.md#networking-requirements)も満たしてください。

vLLM と SGLang DPA/EP の例では `Qwen/Qwen3-30B-A3B`、通常の SGLang DP の例では `Qwen/Qwen3-8B` を使用します。これらはリソース配置を示す例です。利用可能なコンテキスト長と同時実行数は、重みの精度、KV キャッシュ、通信ライブラリ、デバイス性能に依存するため、実環境で確認してください。

## ステップ 1：モデルのデプロイ { #step-1-deploy-the-model }

以下の例では、各デプロイを GPUStack で設定する方法を示します。別の例を試す前に、前のデプロイを停止して GPU を解放してください。

### vLLM External-LB でデプロイ { #deploy-with-vllm-external-lb }

1. `Deployments` で `Deploy Model` をクリックし、`Hugging Face` を選択して `Qwen/Qwen3-30B-A3B` を検索します。
2. `Backend` を `vLLM`、`Backend Version` を `0.32.0` に設定します。
3. `Allow Distributed Inference Across Workers` を有効にします。`Scheduling` で `Scheduling Mode` を `Specify GPU` に設定し、`GPU Selector` で各 Worker の GPU を 4 基ずつ選択します。
4. `Backend Parameters` に以下を追加します。

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

5. `Save` をクリックします。

この配置では `DP × TP = 4 × 2 = 8` 基の GPU を使用し、各 Worker に完全な rank を 2 つ配置します。GPUStack がデバイスとポートを割り当て、各 rank の起動引数を生成し、インスタンス全体の準備が整った後に 4 つの rank API を公開します。

### vLLM Multi-port External-LB でデプロイ { #deploy-with-vllm-multi-port-external-lb }

1. External-LB の例のモデル、バージョン、GPU 選択、その他のパラメーターを使用します。
2. `--data-parallel-external-lb` を次のフラグに置き換えます。

    ```bash
    --data-parallel-multi-port-external-lb
    ```

3. `Save` をクリックします。

公開される rank API は引き続き独立した 4 つです。各 Worker のネイティブ DPSupervisor が 2 つのローカル rank を管理します。集約ヘルスポートは共有推論 API ではありません。

各 rank は 1 台の Worker 内に完全に収まる必要があります。全 Worker の local DP は同じ値で、少なくとも `2` が必要です。`--api-server-count` と `--prefill-context-parallel-size` を明示する場合は `1` にしてください。`--headless`、`--grpc`、`--uds` は使用できません。

### vLLM Hybrid-LB でデプロイ { #deploy-with-vllm-hybrid-lb }

1. External-LB の例のモデル、バージョン、GPU 選択を使用します。
2. `Backend Parameters` のリスト全体を以下に置き換えます。

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=4
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-hybrid-lb
    --max-model-len=8192
    ```

3. `Save` をクリックします。

この例では 2 つの共有 API を公開し、それぞれが 2 つのローカル rank を管理します。ゲートウェイが共有 API を選択し、vLLM がその先の rank を選択します。共有 API のリクエスト数は、個別の rank のリクエスト数ではありません。

通常の External-LB と Hybrid-LB は、Worker をまたぐ TP/PP 分割にも対応します。同じ rank が複数 Worker にまたがる場合、各ノードに同数のデバイスを割り当てる必要があります。この場合、計算だけを担当し API を公開しない Worker もあります。実際のワークロード配置を確認してください。

### SGLang DP と SMG でデプロイ { #deploy-with-sglang-dp-and-smg }

1. `Deployments` で `Deploy Model` をクリックし、`Hugging Face` を選択して `Qwen/Qwen3-8B` を検索します。
2. `Backend` を `SGLang`、`Backend Version` を `0.5.22` に設定します。
3. `Scheduling` で `Scheduling Mode` を `Specify GPU` に設定し、`GPU Selector` で同じ Worker の GPU を 4 基選択します。
4. `Backend Parameters` に以下を追加します。

    ```bash
    --tp-size=1
    --dp-size=4
    --context-length=8192
    --policy=cache_aware
    ```

5. `Save` をクリックします。

通常の DP は完全なモデルコピー 4 つを `DP × TP = 4 × 1 = 4` 基の GPU で実行します。GPUStack が Engine と CPU のみを使用する SMG コンポーネントを作成し、ゲートウェイは SMG に接続します。別途 SMG をデプロイしたり `--worker-urls` を手動設定したりする必要はありません。この例では DP attention を有効にしないため、4 基すべてを同じ Worker に配置してください。

`--policy` は付属の SMG にのみ適用され、Engine には渡されません。既定値は `cache_aware` で、`random`、`round_robin`、`power_of_two`、`manual` も使用できます。同じプレフィックスを持つリクエストは同じターゲットに送られ続けることがあります。分散が均等でないだけでルーティング障害とは判断できません。

### SGLang DPA と EP でデプロイ { #deploy-with-sglang-dpa-and-ep }

1. 前の例と同じ 1 台の Worker の GPU 4 基と SGLang バージョンを使用し、モデルを `Qwen/Qwen3-30B-A3B` に変更します。
2. `Backend Parameters` のリスト全体を以下に置き換えます。

    ```bash
    --tp-size=4
    --dp-size=1
    --attn-dp-size=4
    --ep-size=4
    --context-length=8192
    --policy=cache_aware
    ```

3. `Save` をクリックします。

この例は 4 基の GPU で attention DP と EP を組み合わせます。通常の DP の式 `DP × TP` で 16 基と数えないでください。`--attn-dp-size` と `--attention-data-parallel-size` は別名であり、既知の Runner バージョンでは SGLang `0.5.22` 以降が必要です。旧バージョンでは `--dp-size=4 --enable-dp-attention` を使用します。同じ並列度を重複指定するために両方の構文を組み合わせないでください。

通常の SGLang デプロイの有効なルーティング rank 数は、DP size と attention DP size の積です。新しい DPA 構文では TP size が attention DP size で割り切れる必要があり、両方の DP size を同時に `1` より大きくすることはできません。[SGLang Activation Logic](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide#activation-logic) を参照してください。GPUStack が SMG を作成するのは、ルーティング rank 数が `1` を超える場合だけです。EP のみでルーティング rank 数が `1` の場合は Engine を直接使用し、通常の SMG の `--policy` は設定できません。SGLang PD は独立した PD router を使用し、各ロールに通常デプロイの SMG を追加しません。

### トポロジーの管理を GPUStack に任せる { #let-gpustack-manage-the-topology }

これらの管理対象の例では、インスタンス全体の並列度、モード、GPU 選択を設定します。rank、ノード番号、通信アドレス、ポートは GPUStack が生成します。Worker ごとに別のデプロイを作成したり、`GPUSTACK_SKIP_GPU_COUNT_CHECK` を設定したりする必要はありません。

`--data-parallel-rank`、`--data-parallel-start-rank`、`--data-parallel-address`、`--data-parallel-rpc-port`、`--nnodes`、`--node-rank`、`--host`、`--port` は GPUStack に任せてください。`--data-parallel-size-local` は明示できますが、実際の割り当てと一致させる必要があります。

!!! warning

    vLLM の 3 つの LB フラグは相互排他的です。管理対象の rank API は TCP HTTP を使用し、Engine 側の `--ssl-keyfile` や `--ssl-certfile` に対応しません。クライアント用 TLS はゲートウェイで設定してください。`GPUSTACK_MANUAL_DP=1` は通常の組み込み vLLM バックエンドでの手動 DP 専用で、PD や SGLang には適用されません。変更後はインスタンスの再作成が必要です。手動の手順は [MultiProcessing チュートリアル](running-distributed-vllm-with-multiprocessing.md#step-2-deploy-the-model-with-the-mp-backend) を参照してください。

## ステップ 2：ワークロードと準備状態の確認 { #step-2-check-workloads-and-readiness }

1. `Deployments` でデプロイを展開し、インスタンスの状態を確認します。
2. インスタンス情報の `Workloads / ranks` を開き、Worker、デバイス、rank/component、API の準備状態を確認します。
3. `Running` になるまで待ちます。ダウンロード完了、コンテナー起動、API の準備完了は別々の段階です。
4. 障害時はワークロードのログ入口かインスタンスの `View Logs` を使い、Worker と原因を特定します。ログのエクスポートには `Download Logs` を使用します。

| 例 | 期待される配置 |
| --- | --- |
| vLLM External-LB / Multi-port External-LB | 4 つの rank API |
| vLLM Hybrid-LB | 2 つの共有 API、それぞれが 2 つのローカル rank を管理 |
| SGLang DP | Engine と SMG のコンポーネント。Engine 内に 4 つの論理 rank |
| SGLang DPA/EP | Engine と SMG のコンポーネント。この例では 4 つの論理 rank |

ログとメトリクスは実際の取得元に応じて確認してください。SGLang の論理 DP ビューは共有 Engine のネイティブ `dp_rank` メトリクスをフィルターします。この次元がないメトリクスはデータが表示されない場合があります。Hybrid の共有 API メトリクスや Engine の集約メトリクスに rank 数を掛けないでください。メトリクスの欠落や一時的なリクエスト数ゼロだけでは Engine 障害とは判断できません。

## ステップ 3：Playground での確認 { #step-3-verify-in-playground }

1. 準備が整ったデプロイの `Open in Playground` をクリックします。
2. デプロイしたモデルが選択されていることを確認し、「データ並列を一文で説明してください。」と入力して送信します。
3. 応答が返ることを確認し、ワークロードのログと監視でリクエスト経路を確認します。

## PD 分離との併用 { #use-with-pd-disaggregation }

PD は Prefill と Decode を別々に設定し、リソースを割り当てます。PD router がリクエストと KV 転送を調整します。通常デプロイのゲートウェイから rank への直接ルーティングは、PD サービス全体には適用されません。PD の入口は常に router です。

### 転送方式と配置の確認 { #check-the-transport-and-layout }

| vLLM モード | Router メンバー | 転送方式の制約 |
| --- | --- | --- |
| External-LB | 各 DP rank の主 API | CUDA NIXL、CUDA Mooncake、Ascend Mooncake。CUDA Mooncake では各 rank を 1 台の Worker 内に収める必要があります |
| Multi-port External-LB | ネイティブ DPSupervisor が起動する各 rank の主 API。各 API を DP=1 として登録 | CUDA NIXL のみ。全 Worker で同じ local DP ≥ 2 が必要で、各 rank は 1 台の Worker 内に収めます |
| Hybrid-LB | API を提供する Worker の共有エンドポイント。各エンドポイントをその local DP で登録 | CUDA NIXL、CUDA Mooncake、Ascend Mooncake。CUDA Mooncake の静的 router では全共有 API の local DP を同じにする必要があります |

### vLLM NIXL グループのデプロイ { #deploy-a-vllm-nixl-group }

1. `Qwen/Qwen3-30B-A3B` のデプロイを作成します。`vLLM` と `Backend Version` `0.32.0` を選択し、`PD Disaggregation` を有効にして、`Transport` で `NIXL` を選択します。
2. `Roles` で `Prefill` と `Decode` の `Parameters and environment` を両方とも `Custom` に設定します。各ロールの `Backend Parameters` に以下のリスト全体を入力してください。モデルレベルのリストは空にし、プラットフォーム管理の `Router` を使用します。

    ```bash
    --distributed-executor-backend=mp
    --data-parallel-backend=mp
    --data-parallel-size=2
    --tensor-parallel-size=2
    --enable-expert-parallel
    --data-parallel-external-lb
    --max-model-len=8192
    ```

3. 各ロールの `Resources and scheduling` を `Custom` に設定します。`Specify GPU` で、1 台目の GPU 4 基を Prefill に、2 台目の GPU 4 基を Decode に割り当てます。
4. `Save` をクリックします。Prefill、Decode、Router の準備が整ったら、ステップ 3 と同様に Playground で確認します。

Prefill と Decode はそれぞれ 2 つの rank を持ち、合計 8 基の GPU を使用します。ロールのパラメーターリストはモデルレベルのリスト全体を置き換え、追記はしません。選択したレシピが KV コネクター、ロールのアドレス、接続パラメーターを生成するため、手動で対向アドレスを指定する必要はありません。

この CUDA NIXL の例では、両ロールの `--data-parallel-external-lb` を `--data-parallel-multi-port-external-lb` に置き換えると、各側に独立した rank API が 2 つずつ残ります。代わりに `--data-parallel-hybrid-lb` に置き換え、各ロールに `--data-parallel-size-local=2` を追加すると、各側が 2 つの rank を管理する共有 API を 1 つ公開します。同時に指定する LB フラグは 1 つだけにしてください。

両側に利用可能なメンバーが揃うと Router が起動します。起動中の `Partially ready` は正常です。rank やロールが 1 つ準備完了になっただけでは PD の全リクエスト経路を確認できません。CUDA Mooncake は静的メンバーシップを使用するため、メンバー変更時に router を再作成し、リクエストを一時中断する場合があります。ロール管理と KV 転送は [PD 分離](../user-guide/pd-disaggregation.md) を参照してください。SGLang PD のルーティングは Router の `Route arguments` で設定し、通常の SMG の例の Engine パラメーター `--policy` は使用しません。

## ステップ 4：デプロイの更新と復旧 { #step-4-update-and-recover-the-deployment }

### 設定変更の適用 { #apply-a-configuration-change }

並列度、LB モード、GPU 配置を変更した場合は、インスタンス全体の再作成が必要です。

1. 新しい設定を保存します。
2. デプロイの `Operations` で `Stop` を選択します。
3. 古いインスタンスが削除されるまで待ちます。PD の停止処理は、オンラインの Worker が古いワークロードの停止を確認するまで待機します。
4. `Start` を選択し、新しいインスタンスの準備が整ったら Playground で確認します。通常のデプロイでスケジュールに基づくスケーリングを有効にしている場合は、`Resume Schedule` で現在のスケジュールを再開します。目標数がゼロのままになる場合もあります。

| 状況 | 復旧動作 |
| --- | --- |
| 通常の管理対象インスタンスで Engine/rank が故障 | インスタンスの全エンドポイントを撤去します。古いワークロードの停止確認後、`Auto-Restart On Error` に従ってインスタンス全体を復旧します。通常のインスタンスは元の Worker/GPU 割り当てで現在の設定を使用して復旧できます。 |
| 通常の SGLang デプロイで SMG のみ故障 | 正常な Engine を維持し、SMG のみ再作成します。 |
| 設定変更なしで PD の Engine インスタンスが故障 | 故障したインスタンス全体を復旧し、正常な対向ロールと他のインスタンスを維持して router のメンバーを更新します。 |
| PD の設定を変更済み | 古い設定のメンバーの自動復旧と再作成を一時停止します。デプロイ全体を停止して開始し、保存済みの設定を適用します。 |
| Worker が到達不能で古いワークロードの状態が不明 | 古いワークロードの管理上の帰属を保持して停止確認を待ちます。別 Worker への自動移動は保証されません。 |

設定を保存しても正常なプロセスは直ちに再起動しません。通常の障害復旧で GPU を再選択することもありません。新しい設定でリソース配置を変える必要がある場合は、復旧による再スケジューリングに頼らず、デプロイを明示的に再作成してください。

### 利用できない Worker の置き換え { #replace-an-unavailable-worker }

Worker が到達不能の場合、GPUStack はそのノードやワークロードの状態を現時点で確認できません。その状態だけでは修復可能かどうかは分かりません。ノードやネットワークを調査し、Worker の交換が必要だと判断した場合は、次の手順に従ってください。

1. デプロイの `Stop` または `Delete` を選択します。GPUStack は到達不能、ハートビート期限切れ、または削除済みと判定された Worker の古いインスタンス記録を削除し、オンラインの Worker が古いワークロードを停止するまで待ちます。
2. デプロイを保持する場合は、代わりの Worker を登録し、`GPU Selector` で GPU 選択を更新します。
3. 古いインスタンスが削除された後に `Start` を選択します。デプロイを削除した場合は、改めて作成します。

!!! warning

    到達不能な Worker のインスタンス記録を削除しても、リモートプロセスが停止したことにはなりません。元のマシンが稼働中でネットワークから隔離されている場合、代わりのグループを起動する前に、その古いワークロードをマシンまたはインフラ側で停止または隔離してください。元の Worker は再接続時に、有効なインスタンスに属さなくなった古いワークロードを削除します。復旧では KV キャッシュや処理中リクエストの保持は保証されません。

## 参考資料 { #references }

- [vLLM 0.32.0: Data Parallel Deployment](https://docs.vllm.ai/en/v0.32.0/serving/data_parallel_deployment/)
- [SGLang: DP, DPA and SGLang DP Router](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
