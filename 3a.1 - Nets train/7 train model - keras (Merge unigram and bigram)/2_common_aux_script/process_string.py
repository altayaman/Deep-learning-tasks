def process_string(sentence_):
    printable_ = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    sentence_processed = "".join((char if char in printable_ else "") for char in sentence_.lower())
    
    return sentence_processed