import { COLORS, createColorExample } from "./colorData.js";

const TRAINING_EXAMPLES_PER_COLOR = 40;
const TRAINING_OPTIONS = { epochs: 40, batchSize: 12 };

function createTrainingExamples() {
  return COLORS.flatMap((color) =>
    Array.from({ length: TRAINING_EXAMPLES_PER_COLOR }, () =>
      createColorExample(color),
    ),
  );
}

export default class ML {
  async trainModel(updateProgress) {
    updateProgress("Initializing TensorFlow...");
    await ml5.tf.ready();

    const trainingExamples = createTrainingExamples();

    updateProgress(`Created ${trainingExamples.length} training examples.`);

    const model = ml5.neuralNetwork({
      inputs: ["red", "green", "blue"],
      outputs: ["label"],
      task: "classification",
      debug: false,
    });

    for (const { red, green, blue, label } of trainingExamples) {
      model.addData({ red, green, blue }, { label });
    }

    updateProgress("Normalizing the training data...");
    model.normalizeData();

    updateProgress("Training the color classifier...");
    await new Promise((resolve, reject) => {
      model.train(
        TRAINING_OPTIONS,
        (epoch) => {
          updateProgress(
            `Training epoch ${epoch + 1} of ${TRAINING_OPTIONS.epochs}...`,
          );
        },
        (_result, error) => {
          if (error) {
            reject(error);
            return;
          }

          resolve();
        },
      );
    });

    this.model = model;
    updateProgress("Training complete.");
  }

  async classify(color) {
    if (!this.model) {
      throw new Error("The color classifier has not been trained yet.");
    }

    const results = await this.model.classify({
      red: color.red,
      green: color.green,
      blue: color.blue,
    });

    const label = results?.[0]?.label;

    if (typeof label !== "string") {
      throw new Error(
        "The color classifier did not return a valid prediction.",
      );
    }

    return label.toLowerCase();
  }
}
