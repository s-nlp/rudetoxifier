def find_bpe_position_by_offset(bpe_offsets, target_offset):
    bpe_nums=[]
    for sent_num, sent in enumerate(bpe_offsets):
        if sent[-1][0] < target_offset[0]:
            continue
        
        for bpe_num, bpe in enumerate(sent):
            if target_offset[0] <= bpe[0] and bpe[1] <= target_offset[1]:
                bpe_nums.append(bpe_num)
        return (sent_num, bpe_nums)
    
    
def generate_seq_indexes(indexes):
    if not indexes:
        yield []
        return

    for ind in indexes[0]:
        for seq in generate_seq_indexes(indexes[1:]):
            yield [ind] + seq
            
"""failure case of tokenizer:
    tagged_text = "Earlier this year , some 70 U.S. congressmen sent a letter to U. __S.__ President Bill Clinton , calling for an end to the humanitarian crisis in Iraq by having the sanctions lifted ."
"""
