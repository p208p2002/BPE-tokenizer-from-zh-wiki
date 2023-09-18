import sentencepiece as spm
import os

# --input（逗號分隔的輸入句子列表） 類型：std::string 預設值：""
# --input_format（輸入格式。支援的格式為 text 或 tsv。） 類型：std::string 預設值：""
# --model_prefix（輸出模型前綴） 類型：std::string 預設值：""
# --model_type（模型算法：unigram、bpe、word 或 char） 類型：std::string 預設值："unigram"
# --vocab_size（詞彙表大小） 類型：int32 預設值：8000
# --accept_language（逗號分隔的此模型可接受的語言列表） 類型：std::string 預設值：""
# --self_test_sample_size（自我測試樣本的大小） 類型：int32 預設值：0
# --character_coverage（字符覆蓋率，用於確定最小符號） 類型：double 預設值：0.9995
# --input_sentence_size（訓練器加載的句子的最大大小） 類型：std::uint64_t 預設值：0
# --shuffle_input_sentence（提前隨機抽樣輸入句子。當 --input_sentence_size > 0 時有效） 類型：bool 預設值：true
# --seed_sentencepiece_size（種子 sentencepieces 的大小） 類型：int32 預設值：1000000
# --shrinking_factor（根據損失保留前 shrinking_factor 個 piece） 類型：double 預設值：0.75
# --num_threads（訓練的執行緒數） 類型：int32 預設值：16
# --num_sub_iterations（EM 子迭代的數量） 類型：int32 預設值：2
# --max_sentencepiece_length（句子 piece 的最大長度） 類型：int32 預設值：16
# --max_sentence_length（句子的最大長度，以位元組計） 類型：int32 預設值：4192
# --split_by_unicode_script（使用 Unicode 腳本來分割句子 piece） 類型：bool 預設值：true
# --split_by_number（按數字（0-9）分割標記） 類型：bool 預設值：true
# --split_by_whitespace（使用空白來分割句子 piece） 類型：bool 預設值：true
# --split_digits（將所有數字（0-9）分成單獨的 piece） 類型：bool 預設值：false
# --treat_whitespace_as_suffix（將空格標記視為後綴而不是前綴。） 類型：bool 預設值：false
# --allow_whitespace_only_pieces（允許僅包含（連續的）空格標記的 piece） 類型：bool 預設值：false
# --control_symbols（逗號分隔的控制符號列表） 類型：std::string 預設值：""
# --control_symbols_file（從文件中加載控制符號。） 類型：std::string 預設值：""
# --user_defined_symbols（逗號分隔的用戶定義符號列表） 類型：std::string 預設值：""
# --user_defined_symbols_file（從文件中加載用戶定義符號。） 類型：std::string 預設值：""
# --required_chars（UTF8 字符，在此標誌中的字符始終在字符集中使用，不受 --character_coverage 影響） 類型：std::string 預設值：""
# --required_chars_file（從文件中加載 required_chars。） 類型：std::string 預設值：""
# --byte_fallback（將未知 piece 分解為 UTF-8 字節 piece） 類型：bool 預設值：false
# --vocabulary_output_piece_score（在詞彙文件中定義 piece 分數） 類型：bool 預設值：true
# --normalization_rule_name（規範化規則名稱。從 nfkc 或 identity 中選擇） 類型：std::string 預設值："nmt_nfkc"
# --normalization_rule_tsv（規範化規則 TSV 文件。） 類型：std::string 預設值：""
# --denormalization_rule_tsv（反規範化規則 TSV 文件。） 類型：std::string 預設值：""
# --add_dummy_prefix（在文本開頭添加虛擬空格） 類型：bool 預設值：true
# --remove_extra_whitespaces（刪除前綴、後綴和重複的內部空格） 類型：bool 預設值：true
# --hard_vocab_limit（如果設置為 false，則 --vocab_size 被視為軟限制。） 類型：bool 預設值：true
# --use_all_vocab (If set to true, use all tokens as vocab. Valid for word/char models.)  type: bool default: false
# --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
# --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
# --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
# --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
# --unk_piece (Override UNK (<unk>) piece.)  type: std::string default: "<unk>"
# --bos_piece (Override BOS (<s>) piece.)  type: std::string default: "<s>"
# --eos_piece (Override EOS (</s>) piece.)  type: std::string default: "</s>"
# --pad_piece (Override PAD (<pad>) piece.)  type: std::string default: "<pad>"
# --unk_surface (Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.)  type: std::string default: " ⁇ "
# --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false
# --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
# --enable_differential_privacy (Whether to add DP while training. Currently supported only by UNIGRAM model.)  type: bool default: false
# --differential_privacy_noise_level (Amount of noise to add for DP)  type: float default: 0
# --differential_privacy_clipping_threshold (Threshold for clipping the counts for DP)  type: std::uint64_t default: 0
# --help (show help)  type: bool default: false
# --version (show version)  type: bool default: false
# --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0


spm.SentencePieceTrainer.train(
    input='data/corpus.txt', 
    model_prefix='zh_en_wiki_bpe_model_sp', 
    user_defined_symbols=["，","。","：","？","（","）","「","」","；"],
    max_sentencepiece_length=12,
    split_digits=True,
    model_type="bpe",
    byte_fallback=True,
    train_extremely_large_corpus=True,
    vocab_size=32000
)
