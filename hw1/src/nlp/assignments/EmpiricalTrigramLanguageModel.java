package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class EmpiricalTrigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static final double lambda1 = 0.6;
	static final double lambda2 = 0.3;

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		double trigramCount = trigramCounter.getCount(prePreviousWord
				+ previousWord, word);
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			unigramCount = wordCounter.getCount(UNKNOWN);
		}
		//Stupid backoff
		double prob = trigramCount;
		if (trigramCount == 0) {
            if (bigramCount == 0) {
                prob = unigramCount;
            } else {
                prob = bigramCount;
            }
        }
        if (trigramCount == 0 && bigramCount ==0 ) {
            //System.out.println("Unseen sequence " + prePreviousWord + previousWord + word);
        }
        //Stupid backoff
        //return  prob;
		return lambda1 * trigramCount + lambda2 * bigramCount + (1- lambda1 - lambda2) * unigramCount;
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePreviousWord = stoppedSentence.get(0);
		String previousWord = stoppedSentence.get(1);
		for (int i = 2; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);

			probability *= getTrigramProbability(prePreviousWord, previousWord,
					word);
			prePreviousWord = previousWord;
			previousWord = word;
		}
		if (probability == 0) {
            System.out.println("Zero probability");
        }
		return probability;
	}

	String generateWord(String prePrevious, String previous) {
		double sample = Math.random();
		double sum = 0.0;
        Counter<String> possibleCurrents = trigramCounter.getCounter(prePrevious + previous);

        for (String word : possibleCurrents.keySet()) {
			sum += possibleCurrents.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord(START, START);
        String prePrevious= START;
		while (!word.equals(STOP)) {
			sentence.add(word);
            String temp = word;
			word = generateWord(prePrevious, word);
            prePrevious = temp;
		}
		return sentence;
	}

	public EmpiricalTrigramLanguageModel(
            Collection<List<String>> trainingSentenceCollection, Collection<List<String>> validationSentenceCollection) {
		updateCounters(trainingSentenceCollection);
        //updateCounters(validationSentenceCollection);
		wordCounter.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	public void updateCounters(Collection<List<String>> sentenceCollection) {
        for (List<String> sentence : sentenceCollection) {
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, START);
            stoppedSentence.add(0, START);
            stoppedSentence.add(STOP);
            String prePreviousWord = stoppedSentence.get(0);
            String previousWord = stoppedSentence.get(1);
            bigramCounter.incrementCount(prePreviousWord, previousWord, 1.0);
            wordCounter.incrementCount(prePreviousWord, 1.0);
            wordCounter.incrementCount(previousWord, 1.0);
            for (int i = 2; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                wordCounter.incrementCount(word, 1.0);
                bigramCounter.incrementCount(previousWord, word, 1.0);
                trigramCounter.incrementCount(prePreviousWord + previousWord,
                        word, 1.0);
                prePreviousWord = previousWord;
                previousWord = word;
            }
        }
    }

	private void normalizeDistributions() {
		for (String previousBigram : trigramCounter.keySet()) {
			trigramCounter.getCounter(previousBigram).normalize();
		}
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}

		//trigramCounter.good_turing_normalize();
        //bigramCounter.good_turing_normalize();
		wordCounter.normalize();
	}
}
