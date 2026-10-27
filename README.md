# bbhost 日本語版

Bloodborne を Windows x64 と Linux x86-64 で実行する bbhost の日本語専用版です。日本版 The Old Hunters（CUSA03014）を標準とし、日本語のゲームデータを維持して欧州オンラインへ接続します。日本版オンラインは利用者による動作確認済みです。

設定画面、PC 設定、ゲーム内の追加メニュー、キー設定、プラグインの表示と説明を日本語化し、両 OS で使える日本語フォントを内蔵しています。地域切替プラグインと、タイトル画面からの再接続・プレイログ送信先の修正をビルドに統合しています。

ゲームデータ、復号済み eboot、セーブ、認証情報は含みません。自分で用意した更新 1.09 適用済みのゲームと、対応する復号済み ELF が必要です。

- [日本語の導入・操作・ビルド手順](docs/japanese-release.md)
- [提供された互換パック全体](compat/jp-eu-v020/README.md)
- [上流の技術概要（英語、参考資料）](README.upstream.md)
- [ライセンス](LICENSE)

Linux のビルド:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j4
./build/bbhost --setup
```

Windows のクロスビルドと両 OS の配布物作成は[日本語手順](docs/japanese-release.md#ビルド)を参照してください。GitHub Actions の既存の Linux / Windows ビルドにも同じ日本語化・互換修正が含まれます。

`compat/jp-eu-v020/` は提供物の原本をチェックサム付きで維持しています。原本作成時の検証状況は履歴であり、現在の利用者による日本版オンラインの動作確認は上記手順に記載しています。
