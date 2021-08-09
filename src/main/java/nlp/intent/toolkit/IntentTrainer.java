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

    public static void main(String[] args) throws Exception {

        File trainingDirectory = new File(args[0]);
        String lang = trainingDirectory.getPath().substring(trainingDirectory.getPath().lastIndexOf('/') + 1);
        // String[] slots = new String[0];
		Map<String, String> doccatAndSlotsMap = new HashMap<String, String>();
        if (args.length > 1) {
            // slots = args[1].split(",");
            for(String keyValue : args[1].split(",")) {
                String[] pairs = keyValue.split("=", 2);
                doccatAndSlotsMap.put(pairs[0], pairs[1]);
            }
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

        // List<TokenNameFinderModel> tokenNameFinderModels = new ArrayList<TokenNameFinderModel>();
            // for (String slot : slots) {
            //     List<ObjectStream<NameSample>> nameStreams = new ArrayList<ObjectStream<NameSample>>();
            //     for (File trainingFile : trainingDirectory.listFiles()) {
            //         ObjectStream<String> lineStream = new PlainTextByLineStream(new MarkableFileInputStreamFactory(trainingFile), "UTF-8");
            //         ObjectStream<NameSample> nameSampleStream = new NameSampleDataStream(lineStream);
            //         nameStreams.add(nameSampleStream);
            //     }
            //     ObjectStream<NameSample>[] n = new NameSampleDataStream[nameStreams.size()];
            //     ObjectStream<NameSample> combinedNameSampleStream = ObjectStreamUtils.createObjectStream(nameStreams.toArray(n));
            //     TokenNameFinderModel tokenNameFinderModel = NameFinderME.train(lang, slot, combinedNameSampleStream, nameFinderTrainingParams, new TokenNameFinderFactory(
            //             readFile("/home/patricia/dev/nlp-intent-toolkit/features.xml"), Collections.emptyMap(), new BioCodec()
            //     ));
            //     combinedNameSampleStream.close();
        //     tokenNameFinderModels.add(tokenNameFinderModel);

        Map<String, TokenNameFinderModel> tokenNameFinderModels = new HashMap<String, TokenNameFinderModel>();

        for (Map.Entry<String, String> entry : doccatAndSlotsMap.entrySet()) {
            String trainingFile = entry.getKey();
            String slot = entry.getValue();
            ObjectStream<String> lineStream = new PlainTextByLineStream(new MarkableFileInputStreamFactory(
                            new File(trainingDirectory + "/" + trainingFile + ".txt")), "UTF-8");
            ObjectStream<NameSample> nameSampleStream = new NameSampleDataStream(lineStream);

            TokenNameFinderModel tokenNameFinderModel = NameFinderME.train(lang, slot, nameSampleStream, nameFinderTrainingParams, new TokenNameFinderFactory(
                            readFile("/home/patricia/dev/nlp-intent-toolkit/features.xml"), Collections.emptyMap(), new BioCodec()
            ));
            nameSampleStream.close();
            tokenNameFinderModels.put(slot, tokenNameFinderModel);

            modelOut = new BufferedOutputStream(new FileOutputStream("./models/namefinders/models-" + lang + "/" + slot + ".bin"));
            tokenNameFinderModel.serialize(modelOut);
        }


        DocumentCategorizerME categorizer = new DocumentCategorizerME(doccatModel);

        Map<String, NameFinderME> nameFinderMEs = new HashMap<String, NameFinderME>();
        for (Map.Entry<String, TokenNameFinderModel> entry : tokenNameFinderModels.entrySet()) {
                nameFinderMEs.put(entry.getKey(), new NameFinderME(entry.getValue()));
        }

        // NameFinderME[] nameFinderMEs = new NameFinderME[tokenNameFinderModels.size()];
        // for (int i = 0; i < tokenNameFinderModels.size(); i++) {
        //   nameFinderMEs[i] = new NameFinderME(tokenNameFinderModels.get(i));
        // }

        System.out.println("Training complete. Ready.");
        System.out.print(">");
        String s;

        InputStream modelIn = new FileInputStream("./models/" + lang + "-token.bin");
        TokenizerModel model = new TokenizerModel(modelIn);
        Tokenizer tokenizer = new TokenizerME(model);

        // while ((s = System.console().readLine()) != null) {
        //   double[] outcome = categorizer.categorize(tokenizer.tokenize(s));
        //   System.out.print("{ action: '" + categorizer.getBestCategory(outcome) + "', args: { ");
        //   String[] tokens = tokenizer.tokenize(s);
        //   for (NameFinderME nameFinderME : nameFinderMEs) {
        //     Span[] spans = nameFinderME.find(tokens);
        //     String[] names = Span.spansToStrings(spans, tokens);
        //     for (int i = 0; i < spans.length; i++) {
        //       if(i > 0) { System.out.print(", "); }
        //       System.out.print(spans[i].getType() + ": '" + names[i] + "' ");
        //     }
        //   }
        //   System.out.println("} }");
        //   System.out.print(">");
        // }

        while ((s = System.console().readLine()) != null) {
            String[] tokens = tokenizer.tokenize(s);
            Map<String, Double> scoreMap = categorizer.scoreMap(tokens);
            if (scoreMap.values().stream().distinct().count() <= 1) {
                String result = "NavigationIntent";
                System.out.print("{ action: '" + result  + "', args: { ");

                try {
                    NameFinderME nameFinderME = nameFinderMEs.get(doccatAndSlotsMap.get(result));
                    Span[] spans = nameFinderME.find(tokens);
                    String[] names = Span.spansToStrings(spans, tokens);
										System.out.print(spans[spans.length - 1].getType() + ": '" + names[names.length - 1] + "' ");
                }
                catch (NullPointerException e) { }
                System.out.println("} }");
            } else {
                double[] outcome = categorizer.categorize(tokenizer.tokenize(s));
                String result = categorizer.getBestCategory(outcome);
                System.out.print("{ action: '" + result  + "', args: { ");

                try {
                    NameFinderME nameFinderME = nameFinderMEs.get(doccatAndSlotsMap.get(result));
                    Span[] spans = nameFinderME.find(tokens);
                    String[] names = Span.spansToStrings(spans, tokens);
                    // add most likely target if more than one is available
                    if (spans.length >= 1) {
                        double[] probs = nameFinderME.probs(spans);
                        double maxProb = Arrays.stream(probs).boxed().max(Double::compareTo).get();
                        int maxIndex = Arrays.asList(Arrays.stream(probs).boxed().toArray(Double[]::new)).indexOf(maxProb);
                        System.out.print(spans[maxIndex].getType() + ": '" + names[maxIndex] + "' ");
                    }
                }
                catch (NullPointerException e) {}
                System.out.println("} }");
            }
            System.out.print(">");
        }
    }
}
