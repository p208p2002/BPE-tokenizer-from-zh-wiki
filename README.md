# Train a BPE tokenizer from zh-wiki
從中文維基百科訓練 BPE Tokenizer。

### 訓練
安裝依賴後，依照編號執行即可。
> 需要54GB以上的記憶體

### 下載
見 [Release](https://github.com/p208p2002/BPE-tokenizer-from-zh-wiki/releases)

### 與其他 tokenizer 比較

#### 測試資料
尚-雅克·盧梭（法語：Jean-Jacques Rousseau，法語發音：[ʒɑ̃ ʒak ʁuso]；1712年6月28日—1778年7月2日）是啟蒙時代的法國與日內瓦哲學家、政治理論家、文學家和音樂家。
盧梭的小說作品《愛彌兒》（Émile）是一篇關於全人公民教育的哲學論文，對康德影響甚大。其言情小說《新愛洛伊斯》對前浪漫主義（pre-romanticism）[19]及浪漫主義時期的小說發展十分重要[20]。
不過，一些知名學者認為盧梭雖然預示了浪漫主義的誕生，但是其「現代文學姿態」其實早已「超越了感傷的浪漫主義」，而其嶄新的語言觀甚至「一直延續到了超現實主義那裡」[21]。

##### zh-wiki BPE tokeinzer
```
▁ <0x0A> ▁ ▁ ▁ ▁尚 - 雅克 · 盧 梭 （ 法語 ： Jean - Jac ques ▁R ous se au ， 法語 發音 ： [ <0xCA> <0x92> ɑ ̃ ▁ <0xCA> <0x92> ak ▁ ʁ us o ] <0xEF> <0xBC> <0x9B> 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 啟蒙 時代的 法國 與 日內瓦 哲學家 、 政治 理論 家 、 文學家 和 音樂家 。 <0x0A> ▁ ▁ ▁ ▁盧 梭 的小說 作品 《 愛 彌 兒 》 （ É m ile ） 是一 篇 關於 全 人 公民 教育的 哲學 論文 ， 對 康德 影響 甚 大 。 其 言 情 小說 《 新 愛 洛 伊斯 》 對 前 浪漫 主義 （ pre - rom ant ic ism ） [ 1 9 ] 及 浪漫 主義 時期的 小說 發展 十分 重要 [ 2 0 ] 。 <0x0A> ▁ ▁ ▁ ▁不過 ， 一些 知名 學者認為 盧 梭 雖然 預 示 了 浪漫 主義的 誕生 ， 但是 其 「 現代 文學 姿態 」 其實 早已 「 超越了 感 傷 的 浪漫 主義 」 ， 而其 嶄 新的 語言 觀 甚至 「 一直 延續 到了 超 現實 主義 那裡 」 [ 2 1 ] 。 <0x0A> ▁ ▁ ▁ ▁
```
> token_count:215

> unk_count:0

##### Wenzhong BPE tokeinzer
Wenzhong模型雖然使用中文語料訓練，可是並沒有針對中文語料建立模型詞表，
雖然依靠BPE演算法可使其 back-off bytes (避免oov)，但因編碼長度變長導致效率較差，並且在令牌化後缺失語義。
```
Ċ Ġ Ġ Ġ Ġå ° ļ - éĽ ħ åħ ĭ Â· çĽ § æ ¢ Ń ï ¼ Ī æ³ ķ èª ŀ ï ¼ ļ Jean - Jac ques ĠRousse au ï ¼ Į æ³ ķ èª ŀ ç Ļ ¼ é Ł ³ ï ¼ ļ [ Ê Ĵ É ĳ Ì ĥ Ġ Ê Ĵ ak Ġ Ê ģ us o ] ï ¼ Ľ 17 12 å¹ ´ 6 æľ Ī 28 æĹ ¥ âĢĶ 17 78 å¹ ´ 7 æľ Ī 2 æĹ ¥ ï ¼ ī æĺ¯ å ķ Ł è Ĵ Ļ æ ĻĤ ä»£ çļĦ æ³ ķ åľ ĭ èĪ ĩ æĹ ¥ åħ § ç ĵ ¦ å ĵ ² åŃ ¸ å® ¶ ãĢģ æ Ķ ¿ æ ² » çĲ Ĩ è « ĸ å® ¶ ãĢģ æĸ ĩ åŃ ¸ å® ¶ å Ĵ Į é Ł ³ æ ¨ Ĥ å® ¶ ãĢĤ Ċ Ġ Ġ Ġ Ġç Ľ § æ ¢ Ń çļĦ å° ı èª ª ä½ľ å ĵ ģ ãĢ Ĭ æĦ Ľ å½ Į åħ Ĵ ãĢ ĭ ï ¼ Ī Ãī mile ï ¼ ī æĺ¯ ä¸Ģ ç ¯ ĩ éĹ ľ æĸ ¼ åħ ¨ äºº åħ ¬ æ° ĳ æķ Ļ è Ĥ ² çļĦ å ĵ ² åŃ ¸ è « ĸ æĸ ĩ ï ¼ Į å° į åº · å¾ · å½ ± é Ł ¿ çĶ ļ å¤§ ãĢĤ åħ ¶ è ¨ Ģ æĥ ħ å° ı èª ª ãĢ Ĭ æĸ ° æĦ Ľ æ ´ Ľ ä¼ Ĭ æĸ ¯ ãĢ ĭ å° į åī į æµ ª æ ¼ « ä¸ » ç ¾ © ï ¼ Ī pre - rom antic ism ï ¼ ī [ 19 ] åı Ĭ æµ ª æ ¼ « ä¸ » ç ¾ © æ ĻĤ æľ Ł çļĦ å° ı èª ª ç Ļ ¼ å ± ķ åį ģ åĪ Ĩ éĩ į è¦ ģ [ 20 ] ãĢĤ Ċ Ġ Ġ Ġ Ġ ä¸į éģ İ ï ¼ Į ä¸Ģ äº Ľ ç Ł ¥ åĲ į åŃ ¸ èĢħ èª į ç Ĥ º çĽ § æ ¢ Ń éĽ ĸ çĦ ¶ é ł Ĳ ç ¤ º äº Ĩ æµ ª æ ¼ « ä¸ » ç ¾ © çļĦ èª ķ çĶŁ ï ¼ Į ä½ Ĩ æĺ¯ åħ ¶ ãĢĮ ç ı ¾ ä»£ æĸ ĩ åŃ ¸ å§ ¿ æ ħĭ ãĢį åħ ¶ å¯ ¦ æĹ © å· ² ãĢĮ è ¶ħ è ¶ Ĭ äº Ĩ æĦ Ł åĤ · çļĦ æµ ª æ ¼ « ä¸ » ç ¾ © ãĢį ï ¼ Į èĢ Į åħ ¶ å ¶ Ħ æĸ ° çļĦ èª ŀ è ¨ Ģ è § Ģ çĶ ļ è ĩ ³ ãĢĮ ä¸Ģ çĽ ´ å » ¶ ç º Į åĪ ° äº Ĩ è ¶ħ ç ı ¾ å¯ ¦ ä¸ » ç ¾ © é Ĥ £ è£ ¡ ãĢį [ 21 ] ãĢĤ Ċ Ġ Ġ Ġ Ġ 
```
> token_count:517

> unk_count:0

##### LLaMA BPE tokeinzer
LLaMA 僅收錄少量中文，大部分中文字仍用 bytes 表示。
```
▁ <0x0A> ▁▁▁▁ <0xE5> <0xB0> <0x9A> - 雅 克 · <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> （ 法 語 ： Jean - Jac ques ▁R ous seau ， 法 語 <0xE7> <0x99> <0xBC> 音 ： [ ʒ ɑ ̃ ▁ ʒ ak ▁ ʁ uso ] ； 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 <0xE5> <0x95> <0x9F> <0xE8> <0x92> <0x99> 時 代 的 法 國 <0xE8> <0x88> <0x87> 日 <0xE5> <0x85> <0xA7> <0xE7> <0x93> <0xA6> <0xE5> <0x93> <0xB2> 學 家 、 政 治 理 論 家 、 文 學 家 和 音 <0xE6> <0xA8> <0x82> 家 。 <0x0A> ▁▁▁▁ <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> 的 小 <0xE8> <0xAA> <0xAA> 作 品 《 愛 <0xE5> <0xBD> <0x8C> <0xE5> <0x85> <0x92> 》 （ É mile ） 是 一 <0xE7> <0xAF> <0x87> <0xE9> <0x97> <0x9C> <0xE6> <0x96> <0xBC> 全 人 公 民 教 育 的 <0xE5> <0x93> <0xB2> 學 論 文 ， <0xE5> <0xB0> <0x8D> 康 德 影 <0xE9> <0x9F> <0xBF> <0xE7> <0x94> <0x9A> 大 。 其 言 情 小 <0xE8> <0xAA> <0xAA> 《 新 愛 <0xE6> <0xB4> <0x9B> 伊 斯 》 <0xE5> <0xB0> <0x8D> 前 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 （ pre - rom antic ism ） [ 1 9 ] 及 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 時 期 的 小 <0xE8> <0xAA> <0xAA> <0xE7> <0x99> <0xBC> 展 十 分 重 要 [ 2 0 ] 。 <0x0A> ▁▁▁▁ 不 <0xE9> <0x81> <0x8E> ， 一 些 知 名 學 者 <0xE8> <0xAA> <0x8D> <0xE7> <0x82> <0xBA> <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> <0xE9> <0x9B> <0x96> 然 <0xE9> <0xA0> <0x90> 示 了 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 的 <0xE8> <0xAA> <0x95> 生 ， <0xE4> <0xBD> <0x86> 是 其 「 現 代 文 學 <0xE5> <0xA7> <0xBF> <0xE6> <0x85> <0x8B> 」 其 <0xE5> <0xAF> <0xA6> <0xE6> <0x97> <0xA9> 已 「 超 越 了 <0xE6> <0x84> <0x9F> <0xE5> <0x82> <0xB7> 的 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 」 ， 而 其 <0xE5> <0xB6> <0x84> 新 的 語 言 <0xE8> <0xA7> <0x80> <0xE7> <0x94> <0x9A> <0xE8> <0x87> <0xB3> 「 一 直 <0xE5> <0xBB> <0xB6> <0xE7> <0xBA> <0x8C> 到 了 超 現 <0xE5> <0xAF> <0xA6> 主 義 那 <0xE8> <0xA3> <0xA1> 」 [ 2 1 ] 。 <0x0A> ▁▁▁▁
```
> token_count:389

> unk_count:0

##### Chinese LLaMA (FlagAlpha/Llama2-Chinese)
同原版LLaMA，沒有修改。
```
▁ <0x0A> ▁▁▁▁ <0xE5> <0xB0> <0x9A> - 雅 克 · <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> （ 法 語 ： Jean - Jac ques ▁R ous seau ， 法 語 <0xE7> <0x99> <0xBC> 音 ： [ ʒ ɑ ̃ ▁ ʒ ak ▁ ʁ uso ] ； 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 <0xE5> <0x95> <0x9F> <0xE8> <0x92> <0x99> 時 代 的 法 國 <0xE8> <0x88> <0x87> 日 <0xE5> <0x85> <0xA7> <0xE7> <0x93> <0xA6> <0xE5> <0x93> <0xB2> 學 家 、 政 治 理 論 家 、 文 學 家 和 音 <0xE6> <0xA8> <0x82> 家 。 <0x0A> ▁▁▁▁ <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> 的 小 <0xE8> <0xAA> <0xAA> 作 品 《 愛 <0xE5> <0xBD> <0x8C> <0xE5> <0x85> <0x92> 》 （ É mile ） 是 一 <0xE7> <0xAF> <0x87> <0xE9> <0x97> <0x9C> <0xE6> <0x96> <0xBC> 全 人 公 民 教 育 的 <0xE5> <0x93> <0xB2> 學 論 文 ， <0xE5> <0xB0> <0x8D> 康 德 影 <0xE9> <0x9F> <0xBF> <0xE7> <0x94> <0x9A> 大 。 其 言 情 小 <0xE8> <0xAA> <0xAA> 《 新 愛 <0xE6> <0xB4> <0x9B> 伊 斯 》 <0xE5> <0xB0> <0x8D> 前 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 （ pre - rom antic ism ） [ 1 9 ] 及 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 時 期 的 小 <0xE8> <0xAA> <0xAA> <0xE7> <0x99> <0xBC> 展 十 分 重 要 [ 2 0 ] 。 <0x0A> ▁▁▁▁ 不 <0xE9> <0x81> <0x8E> ， 一 些 知 名 學 者 <0xE8> <0xAA> <0x8D> <0xE7> <0x82> <0xBA> <0xE7> <0x9B> <0xA7> <0xE6> <0xA2> <0xAD> <0xE9> <0x9B> <0x96> 然 <0xE9> <0xA0> <0x90> 示 了 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 的 <0xE8> <0xAA> <0x95> 生 ， <0xE4> <0xBD> <0x86> 是 其 「 現 代 文 學 <0xE5> <0xA7> <0xBF> <0xE6> <0x85> <0x8B> 」 其 <0xE5> <0xAF> <0xA6> <0xE6> <0x97> <0xA9> 已 「 超 越 了 <0xE6> <0x84> <0x9F> <0xE5> <0x82> <0xB7> 的 <0xE6> <0xB5> <0xAA> <0xE6> <0xBC> <0xAB> 主 義 」 ， 而 其 <0xE5> <0xB6> <0x84> 新 的 語 言 <0xE8> <0xA7> <0x80> <0xE7> <0x94> <0x9A> <0xE8> <0x87> <0xB3> 「 一 直 <0xE5> <0xBB> <0xB6> <0xE7> <0xBA> <0x8C> 到 了 超 現 <0xE5> <0xAF> <0xA6> 主 義 那 <0xE8> <0xA3> <0xA1> 」 [ 2 1 ] 。 <0x0A> ▁▁▁▁ 
```

> token_count:389

> unk_count:0

##### Chinese LLaMA (Linly-AI/Chinese-LLaMA-2)
對詞表添加中文常用字(char level)。
```
▁ <0x0A> ▁▁▁▁ 尚 - 雅 克 · <0xE7> <0x9B> <0xA7> 梭 （ 法 語 ： Jean - Jac ques ▁R ous seau ， 法 語 <0xE7> <0x99> <0xBC> 音 ： [ ʒ ɑ ̃ ▁ ʒ ak ▁ ʁ uso ] ； 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 <0xE5> <0x95> <0x9F> 蒙 時 代 的 法 國 <0xE8> <0x88> <0x87> 日 內 瓦 哲 學 家 、 政 治 理 論 家 、 文 學 家 和 音 <0xE6> <0xA8> <0x82> 家 。 <0x0A> ▁▁▁▁ <0xE7> <0x9B> <0xA7> 梭 的 小 <0xE8> <0xAA> <0xAA> 作 品 《 愛 彌 兒 》 （ É mile ） 是 一 篇 <0xE9> <0x97> <0x9C> 於 全 人 公 民 教 育 的 哲 學 論 文 ， <0xE5> <0xB0> <0x8D> 康 德 影 <0xE9> <0x9F> <0xBF> 甚 大 。 其 言 情 小 <0xE8> <0xAA> <0xAA> 《 新 愛 洛 伊 斯 》 <0xE5> <0xB0> <0x8D> 前 浪 漫 主 義 （ pre - rom antic ism ） [ 1 9 ] 及 浪 漫 主 義 時 期 的 小 <0xE8> <0xAA> <0xAA> <0xE7> <0x99> <0xBC> 展 十 分 重 要 [ 2 0 ] 。 <0x0A> ▁▁▁▁ 不 <0xE9> <0x81> <0x8E> ， 一 些 知 名 學 者 <0xE8> <0xAA> <0x8D> 為 <0xE7> <0x9B> <0xA7> 梭 <0xE9> <0x9B> <0x96> 然 <0xE9> <0xA0> <0x90> 示 了 浪 漫 主 義 的 <0xE8> <0xAA> <0x95> 生 ， 但 是 其 「 現 代 文 學 姿 <0xE6> <0x85> <0x8B> 」 其 <0xE5> <0xAF> <0xA6> 早 已 「 超 越 了 感 <0xE5> <0x82> <0xB7> 的 浪 漫 主 義 」 ， 而 其 嶄 新 的 語 言 <0xE8> <0xA7> <0x80> 甚 至 「 一 直 延 <0xE7> <0xBA> <0x8C> 到 了 超 現 <0xE5> <0xAF> <0xA6> 主 義 那 <0xE8> <0xA3> <0xA1> 」 [ 2 1 ] 。 <0x0A> ▁▁▁▁
```
> token_count:325

> unk_count:0

##### Chinese LLaMA (ziqingyang/chinese-llama-2)
目前唯一有做中文 sentence piece 詞表括擴充的中文 LLaMA。
```
▁ <0x0A> ▁▁▁▁ 尚 - 雅 克 · 盧 梭 （ 法 語 ： Jean - Jac ques ▁R ous seau ， 法 語 發 音 ： [ ʒ ɑ ̃ ▁ ʒ ak ▁ ʁ uso ] ； 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 啟 蒙 時代 的 法國 與 日 內 瓦 哲 學家 、 政治 理 論 家 、 文學 家 和 音樂 家 。 <0x0A> ▁▁▁▁ 盧 梭 的小 說 作品 《 愛 彌 兒 》 （ É mile ） 是 一篇 關 於 全 人 公民 教育 的 哲 學 論 文 ， 對 康 德 影響 甚 大 。 其 言 情 小說 《 新 愛 洛 伊 斯 》 對 前 浪漫 主義 （ pre - rom antic ism ） [ 1 9 ] 及 浪漫 主義 時期 的小 說 發展 十分 重要 [ 2 0 ] 。 <0x0A> ▁▁▁▁ 不過 ， 一些 知名 學 者 認為 盧 梭 雖然 預 示 了 浪漫 主義 的 誕 生 ， 但是 其 「 現代 文學 姿 態 」 其 實 早已 「 超越 了 感 傷 的 浪漫 主義 」 ， 而 其 嶄 新的 語言 觀 甚至 「 一直 延 續 到了 超 現 實 主義 那 裡 」 [ 2 1 ] 。 <0x0A> ▁▁▁▁ 
```
> token_count:229

> unk_count:0

##### Word-level tokenizer
```
尚 - 雅克 · 盧 梭 （ 法語 ： Jean - Jacques R ou s se au ， 法語 發音 ： [ <unk> ɑ <unk> a k <unk> us o ] ； 1712 年 6 月 28 日 — 1778 年 7 月 2 日 ） 是 啟蒙 時代 的 法國 與 日內瓦 哲學家 、 政治 理論 家 、 文學家 和 音樂家 。 盧 梭 的 小說 作品 《 愛 <unk> 兒 》 （ É mi le ） 是 一篇 關於 全 人 公民 教育 的 哲學 論文 ， 對 康德 影響 甚大 。 其 言 情 小說 《 新 愛 洛伊 斯 》 對 前 浪漫 主義 （ p re - ro man ti c is m ） [ 19 ] 及 浪漫 主義 時期 的 小說 發展 十分 重要 [ 20 ] 。 不過 ， 一些 知名 學者 認為 盧 梭 雖然 預 示 了 浪漫 主義的 誕生 ， 但是 其 「 現代 文學 姿態 」 其實 早已 「 超越 了 感 傷 的 浪漫 主義 」 ， 而 其 嶄 新 的 語言 觀 甚至 「 一直 延續 到 了 超 現實 主義 那裡 」 [ 21 ] 。
```
> token_count:192

> unk_count:4

##### Char-level tokenizer
```
尚 - 雅 克 · 盧 梭 （ 法 語 ： J e a n - J a c q u e s R o u s s e a u ， 法 語 發 音 ： [ ʒ ɑ ̃ ʒ a k ʁ u s o ] ； 1 7 1 2 年 6 月 2 8 日 — 1 7 7 8 年 7 月 2 日 ） 是 啟 蒙 時 代 的 法 國 與 日 內 瓦 哲 學 家 、 政 治 理 論 家 、 文 學 家 和 音 樂 家 。 盧 梭 的 小 說 作 品 《 愛 彌 兒 》 （ É m i l e ） 是 一 篇 關 於 全 人 公 民 教 育 的 哲 學 論 文 ， 對 康 德 影 響 甚 大 。 其 言 情 小 說 《 新 愛 洛 伊 斯 》 對 前 浪 漫 主 義 （ p r e - r o m a n t i c i s m ） [ 1 9 ] 及 浪 漫 主 義 時 期 的 小 說 發 展 十 分 重 要 [ 2 0 ] 。 不 過 ， 一 些 知 名 學 者 認 為 盧 梭 雖 然 預 示 了 浪 漫 主 義 的 誕 生 ， 但 是 其 「 現 代 文 學 姿 態 」 其 實 早 已 「 超 越 了 感 傷 的 浪 漫 主 義 」 ， 而 其 嶄 新 的 語 言 觀 甚 至 「 一 直 延 續 到 了 超 現 實 主 義 那 裡 」 [ 2 1 ] 。
```
> token_count:289

> unk_count:0