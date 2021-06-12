import nltk
import json
import heapq
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords

# nltk.download('punkt')
import os


class partial_index:  # will represent a partial index object
    def __init__(self, index_file, posting_file):
        self.indexfile = open(index_file, 'rb')  # binary so that the  input is buffered
        self.postingfile = open(posting_file, 'rb')
        self.currterm = ""  # Current term
        self.offset = 0
        self.next()
        self.finished = False

    def __del__(self):
        self.indexfile.close()
        self.postingfile.close()

    def next(self):  # updates to next term and its offset
        val = self.indexfile.readline().decode().rsplit(',', 1)
        if val[0] == "":
            self.finished = True  # eof reached
            return
        self.currterm = val[0]
        self.offset = int(val[1])

    def retrieve(self):  # will return current term and its posting list and also move to the next term in index list
        self.postingfile.seek(self.offset, 0)  # absolute positioning
        posting = self.postingfile.readline().decode()
        term = self.currterm
        self.next()
        return term, posting


class Merger:
    def __init__(self):
        self.Index_outfile = open("Final_Index.txt", 'wb')
        self.Posting_outfile = open("Final_Posting.txt", 'wb')
        self.offset = 0
        self.PIO = []  # list containing partial index objects

    def __del__(self):
        self.Index_outfile.close()
        self.Posting_outfile.close()

    def addPI(self, partial_index_object):
        self.PIO.append(partial_index_object)

    @staticmethod
    def combine(PIOList):  # combine same terms
        d = None
        term = ""
        for x in PIOList:
            # got same term so need to combine them
            term, plist = x.retrieve()
            dic2 = Inverted_Index.str_to_post(term, plist)
            d = Inverted_Index.combine(term, d, dic2)
        string = Inverted_Index.dict_to_str(d, term)
        return term, string

    def merge(self):
        offset = 0  # the postinglist offset
        # term = None
        # index = 0
        Mergelist = []
        while len(self.PIO) != 0:
            i = 1
            if self.PIO[0].finished is False:
                term = self.PIO[0].currterm
                index = 0
            else:
                self.PIO.pop(0)
                continue
            while i < len(self.PIO):
                # for i in range(1, len(self.PIO)):
                if self.PIO[i].finished is True:
                    self.PIO.pop(i)
                    continue
                elif self.PIO[i].currterm < term:
                    term = self.PIO[i].currterm
                    index = i
                    Mergelist.clear()
                elif self.PIO[i].currterm == term:
                    Mergelist.append(self.PIO[i])
                i += 1

            if len(Mergelist) == 0:
                term, posting = self.PIO[index].retrieve()
            else:
                Mergelist.append(self.PIO[index])
                term, posting = self.combine(Mergelist)
                Mergelist.clear()

            self.Index_outfile.write(term.encode() + ",".encode() + str(offset).encode() + '\n'.encode())
            offset += self.Posting_outfile.write(posting.encode())
        print("Merging Done")


# Will be used to merge individual index into one index (represent a single line)
class entry:
    def __init__(self):
        self.dic = {}

    def __lt__(self, other):
        term_1 = next(iter(self.dic))
        term_2 = next(iter(other.dic))
        if term_1 < term_2:
            return term_1
        elif term_1 > term_2:
            return term_2
        elif term_1 == term_2:
            return term_1

    def create_entry(self, term, string):
        self.dic = Inverted_Index.str_to_post(term, string)


class Inverted_Index:
    def __init__(self):
        self.index = {}  # Dict for index

    def add(self, tokdict):  # dictionary containing tokens for a document
        for term in tokdict:
            ID = tokdict[term]["docID"]
            if term in self.index:
                self.index[term].update({ID: tokdict[term]["poslist"]})  # add a new dictonary
            else:  # create a new entry for the term
                self.index[term] = {ID: tokdict[term]["poslist"]}

    def dumptofile(self, filename):
        termfile = filename + "_term.txt"
        postfile = filename + "_post.txt"
        current_byte = 0
        with open(termfile, 'wb') as t:
            with open(postfile, 'wb') as p:
                for term in sorted(self.index.keys()):
                    tup = term
                    t.write(term.encode('UTF-8', 'replace') + ",".encode('UTF-8', 'replace') + str(current_byte).encode(
                        'UTF-8', 'replace') + '\n'.encode('UTF-8', 'replace'))  # term written
                    pstring = (self.post_to_str(term))
                    current_byte += p.write(pstring.encode('UTF-8', 'replace'))

    def post_to_str(self, term):  # with return optimize delta posting of term in string form
        df = len(self.index[term])
        pstring = str(df) + ","  # df
        prevID = 0
        for docID in self.index[term]:
            pstring += str(docID - prevID) + ","  # gap/ID
            prevID = docID

            freq = len(self.index[term][docID])  # frequency of occurrence in current doc
            pstring += str(freq)  # frequency

            prevOccur = 0
            for o in self.index[term][docID]:
                pstring += "," + str(o - prevOccur)  # positions
                prevOccur = o
            pstring += ","

        pstring = pstring[:-1]
        pstring += "\n"  # frequency
        return pstring

    @staticmethod
    def dict_to_str(dic, term):  # with return optimize delta posting of term in string form
        df = len(dic[term])
        pstring = str(df) + ","  # df
        prevID = 0
        for docID in dic[term]:
            pstring += str(docID - prevID) + ","  # gap/ID
            prevID = docID

            freq = len(dic[term][docID])  # frequency of occurrence in current doc
            pstring += str(freq)  # frequency

            prevOccur = 0
            for o in dic[term][docID]:
                pstring += "," + str(o - prevOccur)  # positions
                prevOccur = o
            pstring += ","

        pstring = pstring[:-1]
        pstring += "\n"  # frequency
        return pstring

    @staticmethod
    def str_to_post(term, string):  # takes a delta optimized string, term and return the original dictionary
        # of term
        numlist = [int(i) for i in string.rsplit(',')]
        IDdict = {}
        j = 0
        df = numlist[j]
        j += 1
        ID = 0
        for _ in range(df):
            ID += numlist[j]
            j += 1
            freq = numlist[j]
            j += 1
            poslist = []
            prev = 0
            for __ in range(freq):
                poslist.append(numlist[j] + prev)
                prev = numlist[j] + prev
                j += 1
            IDdict.update({ID: poslist})

        termdict = {term: IDdict}
        return termdict

    @staticmethod
    def combine(term, dic1,
                dic2) -> object:  # take two dictionaries of same term and merge them, returns new sorted dict of term

        if dic1 is None:
            return dic2

        newdic = dic1.copy()
        for key in dic2[term]:
            newdic[term].update({key: dic2[term][key]})
        # sort the final dic
        sorted_dict = sorted(newdic[term].items(), key=lambda kv: kv[0])
        return {term: dict(sorted_dict)}

    # Preporcessing Part
    def htmlparser(self, filename):
        try:
            tokdict = {}
            with open(filename, 'r', encoding="utf8") as fin:
                for line in fin:
                    if "DOCTYPE" not in line and "<HTML>" not in line:
                        continue
                    else:
                        break
                soup = BeautifulSoup(fin, "html.parser")
                tokstr = soup.get_text("\n", strip=True)
                docID = self.get_Doc_ID()
                # Tokenize
                tok = word_tokenize(tokstr)

                pos = 0
                for word in tok:
                    posting = tokdict.get(word.lower(), None)
                    if posting is None:
                        # Word is not in Dict add it
                        posting = {"docID": docID, "poslist": [pos]}
                        tokdict[word.lower()] = posting
                    else:  # word already exist update positions and frequency
                        posting.get("poslist").append(pos)
                        # posting.get("docID")[] += 1
                    pos += 1
            return tokdict
        except:
            return tokdict

    def apply_stopwords(self, tokdict):
        sw = stopwords.words("english")
        for key in list(tokdict):
            if key in sw:
                tokdict.pop(key)

    def apply_stemming(self, tokdict):  # using snow ball stemmer
        stemmer = nltk.stem.SnowballStemmer('english')
        for key in list(tokdict):
            updated_key = stemmer.stem(key)
            # only update if stemming is actually done
            if key != updated_key:
                # check if that updated key already is exist as many words can have same stemming outputs
                check = tokdict.get(updated_key, None)
                if check is None:  # add the updated key
                    tokdict[updated_key] = tokdict[key]
                    tokdict.pop(key)  # remove the old key
                else:  # updated key is already in dic just append postings and update frequency
                    tokdict[updated_key]["poslist"].extend(tokdict[key]["poslist"])  #
                    tokdict[updated_key]["poslist"].sort()  # sort the dict as well

    def compute_stats(self, tokdict):  # compute lenght and Magnitute
        length = 0
        mag = 0
        k = ''
        for key in tokdict.keys():
            mag += len(tokdict[key]["poslist"]) ** 2
            length += len(tokdict[key]["poslist"]) * len(key)
            k = key
        mag = mag ** 0.5  # square root
        return tokdict[k]["docID"], length, mag

    def get_Doc_ID(self, filename="docInfo.txt"):
        # will return us a DocID for the Passed filename and create its entry in docInfotxt
        with open(filename, 'r') as fin:
            if os.path.getsize("docInfo.txt") == 0:
                return 1
            else:
                new = 1
                for line in fin:
                    data = json.loads(line)
                    if data["DocID"] > new:
                        new = data["DocID"]
            return new + 1

    def save_stats(self, docID, len, mag, path, fileto_save="docInfo.txt"):
        with open(fileto_save, 'a') as fin:
            dirname, filename = os.path.split(path)
            json.dump({"DocID": docID, "DocName": filename, 'directory': dirname, "length": len, "magnitude": mag}, fin)
            fin.write('\n')

    def parse(self, path) -> object:  # will perform complete parsing
        data = self.htmlparser(path)
        if data == {} or data is None:
            return data
        self.apply_stopwords(data)
        self.apply_stemming(data)
        self.save_stats(*self.compute_stats(data), path)
        # docID, length, mag = self.compute_stats(data)
        return data

    def print(self, tokdict):
        print(tokdict.keys())

    def printall(self, tokdict):
        for pair in tokdict.items():
            print(pair)

    # takes a directory and create a partial index for it
    def dir_construct(self, dir):
        pass


# The search engine class

class engine:
    def __init__(self, indexlist, postlist, doclist):
        self.ilist = open(indexlist, 'rb')
        self.plist = open(postlist, 'rb')
        self.doclist = open(doclist, 'rb')

    def __del__(self):
        self.ilist.close()
        self.plist.close()

    def getinfo(self, word):  # will return relevant documents for the term
        self.ilist.seek(0, 0)
        self.doclist.seek(0, 0)
        doclist = []
        for line in self.ilist:
            w = line.decode().rsplit(',', 1)

            if w[0] == word:
                self.plist.seek(int(w[1]), 0)
                string = self.plist.readline()

                dic = Inverted_Index.str_to_post(word, string.decode())

                for key in sorted(dic[word].keys()):
                    for x in self.doclist:
                        data = json.loads(x)
                        if data["DocID"] == key:
                            doclist.append((data["DocID"], data["DocName"], word,data["directory"]))
                            break
                return doclist
        return doclist

    def search(self, query):
        # we need to parse the query first
        sw = stopwords.words("english")
        stemmer = nltk.stem.SnowballStemmer('english')
        tok = word_tokenize(query)  # create token
        for i in range(len(tok)):  # convert to lowercase
            tok[i] = tok[i].lower()
            # apply stop words
        for index, word in enumerate(tok):
            if word in sw:
                tok.pop(index)
            # do stemming
        for i in range(len(tok)):
            tok[i] = stemmer.stem(tok[i])
            # remove duplicates
        tok = list(dict.fromkeys(tok))

        doclist = []
        for t in tok:
            doclist.extend(self.getinfo(t))

        if len(doclist)>0:
            dic = dict((tup[1],tup[3])for tup in doclist)
            #dic = dict((tup[0], [tup[1], tup[3]]) for tup in doclist)
            for key in sorted(dic.keys()):
                #print("DocID: ",key, "DocName: ",dic[key][0], "Dir: ",dic[key][1])
                print(dic[key],key)
        else:
            print("term not Not Found")
        # for tup in doclist:
        #     print("DocID:", tup[0], "DocName:", tup[1], "term", tup[2])




if __name__ == '__main__':
    os.chdir("D:/IR_resources")
    # root = 'D:/IR_resources'
    # for _, dirs, __ in os.walk(root):
    #     for DIR in dirs:
    #         c = Inverted_Index()
    #         for dirpath, _, filename in os.walk(os.path.join(root, DIR)):
    #             for file in filename:
    #                 toks = c.parse(os.path.join(root, DIR, file))
    #                 c.add(toks)
    #         c.dumptofile(str(DIR))

    # c = Preprocess()
    # test1 = c.parse("test1")
    # # c.printall(test1)
    # print()
    # print()
    # test2 = c.parse("test2")
    # # c.printall(test2)
    #
    # dex = Inverted_Index()
    # dex.add(test1)
    # dex.dumptofile("index_1")
    #
    # pex = Inverted_Index()
    # pex.add(test2)
    # pex.dumptofile("index_2")
    #
    # c.printall(dex.index)

    # f1 = partial_index("1_term.txt", "1_post.txt")
    # f2 = partial_index("2_term.txt", "2_post.txt")
    # f3 = partial_index("3_term.txt", "3_post.txt")
    # # #
    # m = Merger()
    # m.addPI(f1)
    # m.addPI(f2)
    # m.addPI(f3)
    # m.merge()

    s = engine("1_term.txt", "1_post.txt", "docInfo.txt")
    s.search("antioxid")



