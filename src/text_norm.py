import html
import re

_MAP = {ord("ي"): "ی", ord("ك"): "ک", ord("ى"): "ی", ord("ـ"): None}
for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _MAP[ord(ch)] = str(i)
for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _MAP[ord(ch)] = str(i)
for cp in range(0x064B, 0x0653):                 
    _MAP[cp] = None
_MAP[0x200B] = " "                          
for cp in (0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E):
    _MAP[cp] = None

_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_REPEAT = re.compile(r"([^\W\d_])\1{2,}")        
_WS = re.compile(r"\u200c*\s+\u200c*")
_ZWNJ_MULTI = re.compile(r"\u200c{2,}")

def normalize(text) -> str:
    if text is None:
        return ""
    t = html.unescape(str(text))
    t = _TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = t.translate(_MAP)
    t = _ZWNJ_MULTI.sub("\u200c", t)
    t = _REPEAT.sub(r"\1\1", t)
    t = _WS.sub(" ", t)
    return t.strip(" \u200c")