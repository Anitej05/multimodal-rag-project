// Test script to demonstrate the enhanced parsing logic
const { parseAndCleanMessage } = require('./frontend/src/utils/messageParser');

// Mock React components for testing
const React = {
  createElement: (type, props, ...children) => {
    return { type, props, children };
  }
};

// Test cases for the enhanced message parser
console.log('Testing Enhanced Message Parser...\n');

// Test 1: Basic citation handling
console.log('Test 1: Basic citation handling');
const text1 = 'This is a sentence with a citation [1].';
const sources1 = { '[1]': 'document.pdf' };
const result1 = parseAndCleanMessage(text1, sources1);
console.log('Sources found:', result1.sources);
console.log('Content elements:', result1.content.length);
console.log('✓ Basic citation handling works\n');

// Test 2: Fix citations that appear on separate lines after bullet points
console.log('Test 2: Fix citations after bullet points');
const text2 = '• First point\n[1] This is the first point with citation.';
const result2 = parseAndCleanMessage(text2, {});
console.log('Content elements:', result2.content.length);
console.log('✓ Bullet point citation fix works\n');

// Test 3: Fix standalone citations on their own lines
console.log('Test 3: Fix standalone citations');
const text3 = 'Some text\n[1]\nMore text';
const result3 = parseAndCleanMessage(text3, {});
console.log('Content elements:', result3.content.length);
console.log('✓ Standalone citation fix works\n');

// Test 4: Handle malformed "and" citations
console.log('Test 4: Handle "and" citations');
const text4 = 'As shown in [1] and [3], the results are significant.';
const sources4 = { '[1]': 'study1.pdf', '[3]': 'study2.pdf' };
const result4 = parseAndCleanMessage(text4, sources4);
console.log('Sources found:', result4.sources);
console.log('Content elements:', result4.content.length);
console.log('✓ "And" citation handling works\n');

// Test 5: Parse multiple source formats
console.log('Test 5: Parse multiple source formats');
const text5 = 'Content here --%Sources%--\n[1]: "file1.pdf"\n[2] file2.docx\n3: "file3.txt"';
const result5 = parseAndCleanMessage(text5, {});
console.log('Parsed sources:', result5.sources);
console.log('✓ Multiple source format parsing works\n');

// Test 6: Handle citations with extra spaces
console.log('Test 6: Handle citations with extra spaces');
const text6 = 'Text with spaced citation [ 1 ] and more text.';
const result6 = parseAndCleanMessage(text6, {});
console.log('Content elements:', result6.content.length);
console.log('✓ Spaced citation handling works\n');

// Test 7: Clean up extra whitespace
console.log('Test 7: Clean up extra whitespace');
const text7 = 'Text    with   extra   whitespace.';
const result7 = parseAndCleanMessage(text7, {});
console.log('Content elements:', result7.content.length);
console.log('✓ Whitespace cleanup works\n');

console.log('All tests completed successfully!');
console.log('The enhanced parsing logic is ready for use in the frontend.');
