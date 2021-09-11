import tf from "@tensorflow/tfjs-node"
import fs from "fs"

const CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz1234567890".split("")
const ENCODING_SIZE = CHARACTER_SET.length
const MAX_WORD_SIZE = 15
const TIME_STEPS = 25

const TAGS = ['WH', 'ADV', 'MOD', 'PRON', 'VERB', 'TO', 'DT', 'ADJ', 'NOUN', 'PREP', 'CONJ', 'NUMB', 'PART', 'AUX']
const TAG_ENCODING_SIZE = TAGS.length

const MAX_BATCH = 1000
const RNN_SIZE = TAG_ENCODING_SIZE * 3
const EPOCHS = 100

const THRESHOLD = 0.0

const MODEL_PATH = `./models/model_${MAX_WORD_SIZE}_${TIME_STEPS}_${RNN_SIZE}_${EPOCHS}`

const oneHotEncodeToken = (token) => {
  const characters = token.toLowerCase().split("")
  const characterIndices = characters.map(c => CHARACTER_SET.indexOf(c))
  characterIndices.length = MAX_WORD_SIZE
  const characterTensor = tf.tensor1d(characterIndices, "int32")
  const encodedCharacters = tf.oneHot(characterTensor, ENCODING_SIZE)
  return encodedCharacters.reshape([MAX_WORD_SIZE * ENCODING_SIZE])
}

const encodeSentence = (sentence) => {
  const tokens = sentence.split(/\s+/)
  return tf.stack(tokens.map(oneHotEncodeToken))
}

const padEncodedSentence = (encodedSentence) => {
  const sentenceLength = encodedSentence.shape[0]
  const zeros = tf.zeros([TIME_STEPS - sentenceLength, MAX_WORD_SIZE * ENCODING_SIZE], "int32")
  return zeros.concat(encodedSentence)
}

const truncateEncodedSentence = (encodedSentence) => {
  return encodedSentence.slice(0, TIME_STEPS)
}

const buildSentenceInputs = (sentence) => {
  const encodedSentence = encodeSentence(sentence)
  const sentenceLength = encodedSentence.shape[0]
  return sentenceLength < TIME_STEPS ? padEncodedSentence(encodedSentence) : truncateEncodedSentence(encodedSentence) 
}

const oneHotEncodeTag = (tag) => {
  const tag_index = TAGS.indexOf(tag)
  return tf.oneHot(tag_index, TAG_ENCODING_SIZE)
}

const encodeTags = (tags) => {
  return tf.stack(tags.map(oneHotEncodeTag))
}

const padEncodedTags = (encodedTags) => {
  const numberOfTags = encodedTags.shape[0]
  const zeros = tf.zeros([TIME_STEPS - numberOfTags, TAG_ENCODING_SIZE], "int32")
  return zeros.concat(encodedTags)
}

const truncateEncodedTags = (encodedTags) => {
  return encodedTags.slice(0, TIME_STEPS)
}

const buildTagOutputs = (tags) => {
  const encodedTags = encodeTags(tags)
  const numberOfTags = encodedTags.shape[0]
  return numberOfTags < TIME_STEPS ? padEncodedTags(encodedTags) : truncateEncodedTags(encodedTags) 
}

const decodeOutput = (predictionTensor) => {
  const predictions = predictionTensor.arraySync()
  return predictions.map((prediction) => {
    const maxIndex = prediction.reduce((max_index, v, i, arr) => (v > arr[max_index] ? i : max_index), 0)
    return prediction[maxIndex] >= THRESHOLD ? TAGS[maxIndex] : "NO TAG"
  })
}

const data = JSON.parse(fs.readFileSync("data/data.json"))
let model = null

if (fs.existsSync(MODEL_PATH)) {
  model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`)
} else {
  const input = tf.input({shape: [TIME_STEPS, MAX_WORD_SIZE * ENCODING_SIZE]})
  
  const rnn = tf.layers.simpleRNN({units: RNN_SIZE, returnSequences: true});
  const rnn_output = rnn.apply(input);

  const denseLayer = tf.layers.dense({units: TAG_ENCODING_SIZE, activation: "softmax"})
  const output = denseLayer.apply(rnn_output)
  
  model = tf.model({inputs: input, outputs: output})
  model.compile({optimizer: 'adam', loss: tf.losses.cosineDistance});
  
  model.summary()
  
  
  for (let i = 0; i < data.length; i += MAX_BATCH) {
    const batch = data.slice(i, i + MAX_BATCH)
    let xs = [], ys = []
  
    batch.forEach(({sentence, tags}, j) => {
      let sentenceInputs = buildSentenceInputs(sentence)
      let tagOutputs = buildTagOutputs(tags)
  
      if (sentenceInputs.shape[0] === tagOutputs.shape[0]) {
        xs.push(sentenceInputs)
        ys.push(tagOutputs)
      } else {
        console.log(sentence, tags)
      }
    })
  
    await model.fit(tf.stack(xs), tf.stack(ys), {
      epochs: EPOCHS,
      verbose: 0,
      callbacks: {
        onTrainEnd: (logs) => { console.log(`Ended training #${i}-${i + batch.length - 1}`) }
      }
    })
  }

  await model.save(`file://${MODEL_PATH}`)
}

const prettyPrintTest = (sentence, outputTags, expectedTags) => {
  const tokens = sentence.split(/\s+/)
  tokens.forEach((token, i) => {
    console.log(token, "=>", outputTags[TIME_STEPS - tokens.length + i], "expected", expectedTags[i])
  })
}

const testModel = (model, sentence, expectedTags) => {
  console.log(`Testing sentence "${sentence}"`)
  const sentenceInputs = buildSentenceInputs(sentence)
  const outputs = model.predict(tf.stack([sentenceInputs]))
  const outputTags = decodeOutput(tf.unstack(outputs)[0])

  prettyPrintTest(sentence, outputTags, expectedTags)
}

testModel(model, data[1].sentence, data[1].tags)


