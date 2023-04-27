import json

def test(f):
	for i, l in enumerate(open(f)):
		try:
			json.loads(l)
		except Exception as e:
			print(f, i, e)


import glob
for f in glob.glob("/home/sbmaruf/data/xCodeEval/retrieval_code_code/train/*file*.jsonl"):
	test(f)

