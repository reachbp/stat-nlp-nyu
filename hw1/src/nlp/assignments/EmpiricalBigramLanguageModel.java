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
class EmpiricalBigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static final double lambda = 0.9;
    double discountFactor = 0.5;
	Counter<String> wordCounter = new Counter<String>();
    CounterMap<String, String> continuationBigram = new CounterMap<String, String>();
    Counter<String> continuationProb = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();

	public double getBigramProbability(String previousWord, String word) {
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount =  wordCounter.getCount(word);

        //Stupid backoff
        double prob = bigramCount;
        if (bigramCount == 0) {
            if (unigramCount == 0) {
                prob = wordCounter.getCount(UNKNOWN);
            } else {
                prob = unigramCount;
            }
        }
        //Stupid backoff
        //return prob;
        if (bigramCount == 0) {
            System.out.println("Context: <"+previousWord+","+word+"> was not seen before. Unigram "+unigramCount);
        }
		return lambda * bigramCount + (1.0 - lambda) * unigramCount;
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

        return continuationProb.getCount(word);
    }

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word);
			previousWord = word;
		}
		if (probability > 1) {
            System.out.println("Something went wrong");
        }
		return probability;
	}

	String generateWord(String previoudWord) {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : bigramCounter.getCounter(previoudWord).keySet()) {
			sum += bigramCounter.getCounter(previoudWord).getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord(START);
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord(word);
		}
		return sentence;
	}

	public EmpiricalBigramLanguageModel(
            Collection<List<String>> trainingsentenceCollection, Collection<List<String>> validationSentenceCollection) {
		updateCounters(trainingsentenceCollection);
        //updateCounters(validationSentenceCollection);
		wordCounter.incrementCount(UNKNOWN, 1.0);
        computeContinuationProbablity();
        //normalizeDistributions();
	}

    private void computeContinuationProbablity() {
        double totalBigrams = bigramCounter.totalModCount();
        for (String previousWord : bigramCounter.keySet()) {
            double totalWordWordCompletes = continuationBigram.getCounter(previousWord).getModCount();
            continuationProb.incrementCount(previousWord, totalWordWordCompletes / totalBigrams);
        }
    }

    public void updateCounters(Collection<List<String>> sentenceCollection) {
        for (List<String> sentence : sentenceCollection) {
            List<String> stoppedSentence = new ArrayList<String>(sentence);
            stoppedSentence.add(0, START);
            stoppedSentence.add(STOP);
            String previousWord = stoppedSentence.get(0);
            wordCounter.incrementCount(START, 1);
            bigramCounter.incrementCount(START, previousWord, 1.0);
            for (int i = 1; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                wordCounter.incrementCount(word, 1.0);
                bigramCounter.incrementCount(previousWord, word, 1.0);
                continuationBigram.incrementCount(word, previousWord, 1.0);
                previousWord = word;
            }
        }
        normalizeDistributions();
    }
	private void normalizeDistributions() {
        /*
		bigramCounter.laplace_normalize();
		wordCounter.laplace_normalize();
		*/
        bigramCounter.good_turing_normalize();
        for (String previousWord : bigramCounter.keySet()) {
            bigramCounter.getCounter(previousWord).good_turing_normalize();
        }
        wordCounter.good_turing_normalize();

	}
	public static void main(String[] args) {


    }
}
