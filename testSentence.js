import { createRequire } from 'module';
const require = createRequire(import.meta.url);

import {getModel, testSentence} from "./modelBuilder.js";
import fs from 'fs'

const yargs = require('yargs/yargs')
const { hideBin } = require('yargs/helpers')
const argv = yargs(hideBin(process.argv)).array("t").argv

const rnnSize = argv.rnn
const timeSteps = argv.ts
const epochs = argv.e

const sentence = argv.s
const expectedTags = argv.t

const [model, modelPath] = await getModel({rnnSize, timeSteps, epochs})

const predictedTags = testSentence(model, sentence, expectedTags)

const ratio = expectedTags.reduce((acc, expectedTag, i) => {
  const predictedTag = predictedTags[i]
  return expectedTag === predictedTag ? acc + 1 : acc
}, 0)/expectedTags.length

const results = [`"${sentence}"`, expectedTags.join(" "), predictedTags.slice(0, expectedTags.length).join(" "), ratio.toFixed(2)].join()
console.log(results)
fs.appendFileSync("tests.csv", results + "\n")