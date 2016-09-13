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
class KatzTrigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static final double lambda1 = 0.6;
	static final double lambda2 = 0.25;
	double discountFactor = 10;
	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
    CounterMap<String, String> continuationBigram = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();
    CounterMap<String, String> continuationTrigram = new CounterMap<String, String>();

	public double getTrigramProbability(String prePreviousWord,
			String previousWord, String word) {
		double trigramCount = trigramCounter.getCount(prePreviousWord
				+ previousWord, word);
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}
		return lambda1 * trigramCount + lambda2 * bigramCount
				+ (1.0 - lambda1 - lambda2) * unigramCount;
	}
	public double getKneserNeyProbability(String prePreviousWord, String previousWord, String word) {
        double trigramCount = trigramCounter.getCount(prePreviousWord
                + previousWord, word);
        double highestOrderNormalizedDiscountNum = Math.max((trigramCount - discountFactor), 0) ;
        double highestOrderNormalizedDiscountDen =  trigramCounter.getCounter(prePreviousWord+previousWord).totalCount();
		double higherOrderTerm = highestOrderNormalizedDiscountNum / highestOrderNormalizedDiscountDen;
        double normalizingConstant = (discountFactor / bigramCounter.getCount(prePreviousWord, previousWord)) *
                trigramCounter.getCounter(prePreviousWord + previousWord).getModCount();
       double lowerOrderTerm = normalizingConstant * getKneserNeyBigram(previousWord, word);
        return higherOrderTerm + lowerOrderTerm;
	}
    public double getKneserNeyBigram(String prev, String word) {
        Counter<String> prevCount = bigramCounter
                .getCounter(prev);
        double normalizedDiscount = Math.max(prevCount.getCount(word) - discountFactor, 0) /  (1 +wordCounter.getCount(prev));
        double normalizingConstantNum = discountFactor * bigramCounter.getCounter(prev).getModCount();
        double normalizingConstantDen = 1 + wordCounter.getCount(prev);
        double normalizingConstant= normalizingConstantNum / normalizingConstantDen;

        return normalizedDiscount + normalizingConstant * getContinuationProbability(word);
    }
    public double getContinuationProbability(String word) {
        double totalWordWordCompletes = continuationBigram.getCounter(word).getModCount();
        double totalBigrams = bigramCounter.totalModCount();
        return totalWordWordCompletes / totalBigrams;
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
		if (probability == 0)
			System.err.println("Underflow");
		return probability;
	}

	String generateWord() {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	public KatzTrigramLanguageModel(Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);

			stoppedSentence.add(STOP);
			String prePreviousWord = stoppedSentence.get(0);
			String previousWord = stoppedSentence.get(1);
            wordCounter.incrementCount(START, 2.0);
            bigramCounter.incrementCount(prePreviousWord, previousWord, 1.0);
			for (int i = 2; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
                continuationBigram.incrementCount(word, previousWord, 1.0);
				trigramCounter.incrementCount(prePreviousWord + previousWord,
						word, 1.0);
                continuationTrigram.incrementCount(previousWord + word, prePreviousWord, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		for (String previousBigram : trigramCounter.keySet()) {
			trigramCounter.getCounter(previousBigram).normalize();
		}
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}
}
