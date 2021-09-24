package nlp.intent.toolkit;

import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.NGramFeatureGenerator;
import opennlp.tools.namefind.*;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.ml.AbstractTrainer;
import opennlp.tools.ml.naivebayes.NaiveBayesTrainer;
import opennlp.tools.util.*;
import opennlp.tools.util.featuregen.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;

public class IntentTrainer {
    public static byte[] readFile(String file) throws IOException {
        File f = new File(file);
        byte[] buffer = new byte[(int)f.length()];
        FileInputStream is = new FileInputStream(file);
        is.read(buffer);
        is.close();
        return buffer;
    }

    public static boolean contains(final String[] array, final String v) {
        for (final String e : array)
            if (e == v || v != null && v.equals(e))
                return true;

        return false;
    }

    public static ArrayList<String> getKeyByValue(Map<String, String[]> map, String value) {
        ArrayList<String> results = new ArrayList<String>();
        for (Map.Entry<String, String[]> entry : map.entrySet()) {
            if (contains(entry.getValue(), value)) {
                results.add(entry.getKey());
            }
        }
        return results;
    }

    public static void main(String[] args) throws Exception {

        File trainingDirectory = new File(args[0]);
        String lang = trainingDirectory.getPath().substring(trainingDirectory.getPath().lastIndexOf('/') + 1);
		Map<String, String[]> slotsAndDoccatsMap = new HashMap<String, String[]>();

        if (args.length > 1) {
            // slots = args[1].split(",");
            for(int i=1; i < args.length; i++){
                String keyValue = args[i];
                String[] pairs = keyValue.split("=", 2);
                slotsAndDoccatsMap.put(pairs[0], pairs[1].split(","));
            }
            System.out.println(slotsAndDoccatsMap);
        }

        if (!trainingDirectory.isDirectory()) {
            throw new IllegalArgumentException("TrainingDirectory is not a directory: " + trainingDirectory.getAbsolutePath());
        }

        List<ObjectStream<DocumentSample>> categoryStreams = new ArrayList<ObjectStream<DocumentSample>>();
        for (File trainingFile : trainingDirectory.listFiles()) {
            String intent = trainingFile.getName().replaceFirst("[.][^.]+$", "");
            ObjectStream<String> lineStream = new PlainTextByLineStream(new MarkableFileInputStreamFactory(trainingFile), "UTF-8");
            ObjectStream<DocumentSample> documentSampleStream = new IntentDocumentSampleStream(intent, lineStream);
            categoryStreams.add(documentSampleStream);
        }

        ObjectStream<DocumentSample>[] d = new IntentDocumentSampleStream[categoryStreams.size()];
        ObjectStream<DocumentSample> combinedDocumentSampleStream = ObjectStreamUtils.createObjectStream(categoryStreams.toArray(d));

        TrainingParameters doccatTrainingParams = new TrainingParameters();
        doccatTrainingParams.put(TrainingParameters.ITERATIONS_PARAM, 100+"");
        doccatTrainingParams.put(TrainingParameters.CUTOFF_PARAM, 0+"");
        // trainingParams.put(AbstractTrainer.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE);

        DoccatFactory customFactory = new DoccatFactory(
            new FeatureGenerator[]{
                    new NGramFeatureGenerator(1, 4),
            }
        );
        DoccatModel doccatModel = DocumentCategorizerME.train(lang, combinedDocumentSampleStream, doccatTrainingParams, customFactory);
        combinedDocumentSampleStream.close();

        BufferedOutputStream modelOut;
        modelOut = new BufferedOutputStream(new FileOutputStream("./models/doccats/model-" + lang + ".bin"));
        doccatModel.serialize(modelOut);

        TrainingParameters nameFinderTrainingParams = new TrainingParameters();
        nameFinderTrainingParams.put(TrainingParameters.ITERATIONS_PARAM, 100+"");
        nameFinderTrainingParams.put(TrainingParameters.CUTOFF_PARAM, 0+"");

        Map<String, TokenNameFinderModel> tokenNameFinderModels = new HashMap<String, TokenNameFinderModel>();

        for (String slot : slotsAndDoccatsMap.keySet()) {
            String[] trainingFiles = slotsAndDoccatsMap.get(slot);

            List<ObjectStream<NameSample>> nameStreams = new ArrayList<ObjectStream<NameSample>>();
            for (String trainingFile : trainingFiles) {
                File file = new File(trainingDirectory + "/" + trainingFile + ".txt");
                if (!file.exists()) {
                    continue;
                }
                ObjectStream<String> lineStream = new PlainTextByLineStream(
                    new MarkableFileInputStreamFactory(file), "UTF-8");
                ObjectStream<NameSample> nameSampleStream = new NameSampleDataStream(lineStream);
                nameStreams.add(nameSampleStream);
            }
            if (nameStreams.size() > 0) {
                ObjectStream<NameSample>[] n = new NameSampleDataStream[nameStreams.size()];
                ObjectStream<NameSample> combinedNameSampleStream = ObjectStreamUtils.createObjectStream(nameStreams.toArray(n));

                TokenNameFinderModel tokenNameFinderModel = NameFinderME.train(lang, slot, combinedNameSampleStream, nameFinderTrainingParams,
                        new TokenNameFinderFactory(readFile("/home/patricia/dev/nlp-intent-toolkit/features.xml"), Collections.emptyMap(), new BioCodec()
                        ));
                combinedNameSampleStream.close();

                tokenNameFinderModels.put(slot, tokenNameFinderModel);

                modelOut = new BufferedOutputStream(new FileOutputStream("./models/namefinders/models-" + lang + "/" + slot + ".bin"));
                tokenNameFinderModel.serialize(modelOut);
            }
        }

        DocumentCategorizerME categorizer = new DocumentCategorizerME(doccatModel);

        Map<String, NameFinderME> nameFinderMEs = new HashMap<String, NameFinderME>();
        for (Map.Entry<String, TokenNameFinderModel> entry : tokenNameFinderModels.entrySet()) {
                nameFinderMEs.put(entry.getKey(), new NameFinderME(entry.getValue()));
        }

        System.out.println("Training complete. Ready.");
        System.out.print(">");
        String s;

        InputStream modelIn = new FileInputStream("./models/" + lang + "-token.bin");
        TokenizerModel model = new TokenizerModel(modelIn);
        Tokenizer tokenizer = new TokenizerME(model);

        while ((s = System.console().readLine()) != null) {
            String[] tokens = tokenizer.tokenize(s);
            Map<String, Double> scoreMap = categorizer.scoreMap(tokens);
            String result;
            if (scoreMap.values().stream().distinct().count() <= 1) {
                result = "NavigationIntent";
            } else {
                double[] outcome = categorizer.categorize(tokenizer.tokenize(s));
                result = categorizer.getBestCategory(outcome);
            }

            System.out.print("{ action: '" + result + "', args: { ");
            try {
                ArrayList<String> slotKeys = getKeyByValue(slotsAndDoccatsMap, result);
                for (String key : slotKeys) {
                    NameFinderME nameFinderME = nameFinderMEs.get(key);
                    Span[] spans = nameFinderME.find(tokens);
										nameFinderME.clearAdaptiveData();
                    String[] names = Span.spansToStrings(spans, tokens);
                    // add most likely target if more than one is available
                    if (spans.length >= 1) {
                        double[] probs = nameFinderME.probs(spans);
                        double maxProb = Arrays.stream(probs).boxed().max(Double::compareTo).get();
                        int maxIndex = Arrays.asList(Arrays.stream(probs).boxed().toArray(Double[]::new)).indexOf(maxProb);
                        System.out.print(spans[maxIndex].getType() + ": '" + names[maxIndex] + "' ");
										} else {
											System.out.print("No target");
										}

										// for (int i = 0; i <= spans.length - 1; i++) {
										// 	System.out.print(spans[i].getType() + ": '" + names[i] + "' ");
										// }
                }
                if (result.equals("SetVolumeActionIntent")) {
                    NameFinderME nameFinderME = nameFinderMEs.get("volume_target");
                    Span[] spans = nameFinderME.find(tokens);
										nameFinderME.clearAdaptiveData();
                    String[] names = Span.spansToStrings(spans, tokens);
                    if (spans.length >= 1) {
                        double[] probs = nameFinderME.probs(spans);
                        double maxProb = Arrays.stream(probs).boxed().max(Double::compareTo).get();
                        int maxIndex = Arrays.asList(Arrays.stream(probs).boxed().toArray(Double[]::new)).indexOf(maxProb);
                        System.out.print(spans[maxIndex].getType() + ": '" + names[maxIndex] + "' ");
										} else {
											System.out.print("No target");
										}
										// for (int i = 0; i <= spans.length - 1; i++) {
                    //     System.out.print(spans[i].getType() + ": '" + names[i] + "' ");
										// }
                }
            }
            catch (NullPointerException e) {
                System.out.println(e.toString());
            }
            System.out.println("} }");
            System.out.print(">");
        }
    }
}
