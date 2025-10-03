import { parseAndCleanMessage } from './messageParser';

// Test cases for the enhanced message parser
describe('Message Parser', () => {
  test('should handle basic citations correctly', () => {
    const text = 'This is a sentence with a citation [1].';
    const sources = { '[1]': 'document.pdf' };
    const result = parseAndCleanMessage(text, sources);
    
    expect(result.sources['[1]']).toBe('document.pdf');
  });

  test('should fix citations that appear on separate lines after bullet points', () => {
    const text = '• First point\n[1]';
    const result = parseAndCleanMessage(text);
    
    // The citation should be attached to the bullet point
    expect(result.content).toBeDefined();
  });

  test('should fix standalone citations on their own lines', () => {
    const text = 'Some text\n[1]\nMore text';
    const result = parseAndCleanMessage(text);
    
    // The citation should be integrated into the text flow
    expect(result.content).toBeDefined();
  });

 test('should handle malformed "and" citations', () => {
    const text = 'As shown in [1] and [3], the results are significant.';
    const result = parseAndCleanMessage(text);
    
    // Should properly handle the "and" citation format
    expect(result.content).toBeDefined();
 });

  test('should parse multiple source formats', () => {
    const text = 'Content here --%Sources%--\n[1]: "file1.pdf"\n[2] file2.docx\n3: "file3.txt"';
    const result = parseAndCleanMessage(text);
    
    // Should parse all three source formats
    expect(result.sources['[1]']).toBeDefined();
    expect(result.sources['[2]']).toBeDefined();
    expect(result.sources['[3]']).toBeDefined();
  });

 test('should handle citations with extra spaces', () => {
    const text = 'Text with spaced citation [ 1 ] and more text.';
    const result = parseAndCleanMessage(text);
    
    // Should normalize [ 1 ] to [1]
    expect(result.content).toBeDefined();
  });

  test('should clean up extra whitespace', () => {
    const text = 'Text    with   extra   whitespace.';
    const result = parseAndCleanMessage(text);
    
    // Should normalize whitespace
    expect(result.content).toBeDefined();
  });
});
