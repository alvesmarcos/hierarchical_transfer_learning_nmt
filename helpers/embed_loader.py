from fairseq import options, utils

embed_dict = utils.parse_embedding('generated_embeds_encoder.txt')

print(len(embed_dict))