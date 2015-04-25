

with open('scores.txt','r') as f:
	scores = f.readlines()

clean_scores = []
for row in scores:
	new_row = row.rstrip('\n')
	new_row = new_row.split(' ')
	new_row[2] = float(new_row[2])
	clean_scores.append(new_row)
print 'rows:{}'.format(len(scores))
final_scores = sorted(clean_scores,key=lambda x: -x[2])

def extract_lang(final_scores,lang):
	english = [row for row in final_scores if row[1]==lang]
	for i in range(20):
		try:
			print english[i]
		except:
			continue
	return english

print 'english:'
english = extract_lang(final_scores,'English')
print 'spanish:'
spanish = extract_lang(final_scores,'Spanish')
print 'catalan:'
catalan = extract_lang(final_scores,'Catalan')


