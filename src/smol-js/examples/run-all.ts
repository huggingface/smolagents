/**
 * Run All Examples
 *
 * This script runs all example files in sequence.
 * Each example demonstrates a different capability of smol-js.
 */

import 'dotenv/config';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

const __dirname = path.dirname(new URL(import.meta.url).pathname);

interface ExampleResult {
  name: string;
  success: boolean;
  duration: number;
  error?: string;
}

async function runExample(filename: string): Promise<ExampleResult> {
  const name = path.basename(filename, '.ts');
  const startTime = Date.now();

  return new Promise((resolve) => {
    const child = spawn('npx', ['tsx', filename], {
      cwd: path.dirname(filename),
      stdio: ['pipe', 'inherit', 'inherit'],
      env: { ...process.env },
    });

    child.on('close', (code) => {
      const duration = Date.now() - startTime;

      if (code === 0) {
        resolve({
          name,
          success: true,
          duration,
        });
      } else {
        resolve({
          name,
          success: false,
          duration,
          error: `Process exited with code ${code}`,
        });
      }
    });

    child.on('error', (err) => {
      resolve({
        name,
        success: false,
        duration: Date.now() - startTime,
        error: err.message,
      });
    });
  });
}

async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║             smol-js Examples Runner                      ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  // Check for API key
  if (!process.env.OPENAI_API_KEY && !process.env.OPENROUTER_API_KEY) {
    console.error('Error: No API key found.');
    console.error('Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env file.\n');
    process.exit(1);
  }

  // Find all example files
  const examplesDir = __dirname;
  const files = fs.readdirSync(examplesDir)
    .filter((f) => f.match(/^\d+-.*\.ts$/) && f !== 'run-all.ts')
    .sort()
    .map((f) => path.join(examplesDir, f));

  if (files.length === 0) {
    console.error('No example files found.');
    process.exit(1);
  }

  console.log(`Found ${files.length} example(s) to run:\n`);
  files.forEach((f) => console.log(`  - ${path.basename(f)}`));
  console.log('\n');

  // Ask if user wants to run all or select
  const args = process.argv.slice(2);
  let filesToRun = files;

  if (args.length > 0) {
    // Filter to specific examples if provided
    filesToRun = files.filter((f) =>
      args.some((arg) => path.basename(f).includes(arg))
    );

    if (filesToRun.length === 0) {
      console.error('No matching examples found for:', args.join(', '));
      process.exit(1);
    }
  }

  // Run each example
  const results: ExampleResult[] = [];

  for (const file of filesToRun) {
    console.log('\n' + '═'.repeat(60));
    console.log(`Running: ${path.basename(file)}`);
    console.log('═'.repeat(60) + '\n');

    const result = await runExample(file);
    results.push(result);

    if (!result.success) {
      console.log(`\n❌ ${result.name} failed: ${result.error}`);
    } else {
      console.log(`\n✅ ${result.name} completed in ${(result.duration / 1000).toFixed(2)}s`);
    }
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  console.log('SUMMARY');
  console.log('═'.repeat(60));

  const passed = results.filter((r) => r.success).length;
  const failed = results.filter((r) => !r.success).length;

  console.log(`\nPassed: ${passed}/${results.length}`);
  console.log(`Failed: ${failed}/${results.length}`);
  console.log(`Total time: ${(results.reduce((sum, r) => sum + r.duration, 0) / 1000).toFixed(2)}s\n`);

  if (failed > 0) {
    console.log('Failed examples:');
    results
      .filter((r) => !r.success)
      .forEach((r) => console.log(`  - ${r.name}: ${r.error}`));
  }

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Runner error:', err);
  process.exit(1);
});
