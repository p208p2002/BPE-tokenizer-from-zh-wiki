import sentencepiece as spm
import sys
sp = spm.SentencePieceProcessor()

sp.load(sys.argv[-1]) # load spm.model

text = """
尚-雅克·盧梭（法語：Jean-Jacques Rousseau，法語發音：[ʒɑ̃ ʒak ʁuso]；1712年6月28日—1778年7月2日）是啟蒙時代的法國與日內瓦哲學家、政治理論家、文學家和音樂家。
盧梭的小說作品《愛彌兒》（Émile）是一篇關於全人公民教育的哲學論文，對康德影響甚大。其言情小說《新愛洛伊斯》對前浪漫主義（pre-romanticism）[19]及浪漫主義時期的小說發展十分重要[20]。
不過，一些知名學者認為盧梭雖然預示了浪漫主義的誕生，但是其「現代文學姿態」其實早已「超越了感傷的浪漫主義」，而其嶄新的語言觀甚至「一直延續到了超現實主義那裡」[21]。
"""

# encode: text => id
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))
