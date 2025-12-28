using NUnit.Framework;
using Nivara;
using Nivara.Exceptions;

namespace Nivara.Tests;

[TestFixture]
public class SchemaTests
{
    [Test]
    public void Constructor_WithValidColumns_CreatesSchema()
    {
        // Arrange
        var columns = new[]
        {
            ("Name", typeof(string)),
            ("Age", typeof(int)),
            ("Salary", typeof(double))
        };

        // Act
        var schema = new Schema(columns);

        // Assert
        Assert.That(schema.ColumnNames.Count, Is.EqualTo(3));
        Assert.That(schema.ColumnNames, Contains.Item("Name"));
        Assert.That(schema.ColumnNames, Contains.Item("Age"));
        Assert.That(schema.ColumnNames, Contains.Item("Salary"));
        Assert.That(schema.ColumnTypes["Name"], Is.EqualTo(typeof(string)));
        Assert.That(schema.ColumnTypes["Age"], Is.EqualTo(typeof(int)));
        Assert.That(schema.ColumnTypes["Salary"], Is.EqualTo(typeof(double)));
    }

    [Test]
    public void Constructor_WithNullColumns_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Schema(null!));
    }

    [Test]
    public void Constructor_WithDuplicateColumnNames_ThrowsArgumentException()
    {
        // Arrange
        var columns = new[]
        {
            ("Name", typeof(string)),
            ("name", typeof(int)) // Case-insensitive duplicate
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new Schema(columns));
        Assert.That(ex.Message, Contains.Substring("Duplicate column name"));
    }

    [Test]
    public void Constructor_WithNullOrWhitespaceColumnName_ThrowsArgumentException()
    {
        // Arrange
        var columns = new[]
        {
            ("", typeof(string)),
            ("Age", typeof(int))
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new Schema(columns));
        Assert.That(ex.Message, Contains.Substring("Column names cannot be null or whitespace"));
    }

    [Test]
    public void Constructor_WithNullColumnType_ThrowsArgumentException()
    {
        // Arrange
        var columns = new[]
        {
            ("Name", typeof(string)),
            ("Age", null!)
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => new Schema(columns));
        Assert.That(ex.Message, Contains.Substring("Column type cannot be null"));
    }

    [Test]
    public void HasColumn_WithExistingColumn_ReturnsTrue()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema.HasColumn("Name"), Is.True);
        Assert.That(schema.HasColumn("name"), Is.True); // Case-insensitive
        Assert.That(schema.HasColumn("Age"), Is.True);
    }

    [Test]
    public void HasColumn_WithNonExistingColumn_ReturnsFalse()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        Assert.That(schema.HasColumn("Age"), Is.False);
        Assert.That(schema.HasColumn("NonExistent"), Is.False);
    }

    [Test]
    public void GetColumnType_WithExistingColumn_ReturnsCorrectType()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema.GetColumnType("Name"), Is.EqualTo(typeof(string)));
        Assert.That(schema.GetColumnType("name"), Is.EqualTo(typeof(string))); // Case-insensitive
        Assert.That(schema.GetColumnType("Age"), Is.EqualTo(typeof(int)));
    }

    [Test]
    public void GetColumnType_WithNonExistingColumn_ThrowsColumnNotFoundException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => schema.GetColumnType("Age"));
        Assert.That(ex.ColumnName, Is.EqualTo("Age"));
        Assert.That(ex.AvailableColumns, Contains.Item("Name"));
    }

    [Test]
    public void GetColumnMetadata_WithExistingColumn_ReturnsMetadata()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act
        var metadata = schema.GetColumnMetadata("Name");

        // Assert
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata.IsNullable, Is.True); // Default value
    }

    [Test]
    public void GetColumnMetadata_WithNonExistingColumn_ThrowsColumnNotFoundException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => schema.GetColumnMetadata("Age"));
        Assert.That(ex.ColumnName, Is.EqualTo("Age"));
    }

    [Test]
    public void WithColumn_WithNewColumn_ReturnsNewSchemaWithColumn()
    {
        // Arrange
        var originalSchema = new Schema(new[] { ("Name", typeof(string)) });

        // Act
        var newSchema = originalSchema.WithColumn("Age", typeof(int));

        // Assert
        Assert.That(newSchema.ColumnNames.Count, Is.EqualTo(2));
        Assert.That(newSchema.HasColumn("Name"), Is.True);
        Assert.That(newSchema.HasColumn("Age"), Is.True);
        Assert.That(newSchema.GetColumnType("Age"), Is.EqualTo(typeof(int)));
        
        // Original schema should be unchanged
        Assert.That(originalSchema.ColumnNames.Count, Is.EqualTo(1));
        Assert.That(originalSchema.HasColumn("Age"), Is.False);
    }

    [Test]
    public void WithColumn_WithExistingColumn_ThrowsArgumentException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => schema.WithColumn("Name", typeof(int)));
        Assert.That(ex.Message, Contains.Substring("Column 'Name' already exists"));
    }

    [Test]
    public void WithColumn_WithCustomMetadata_AddsColumnWithMetadata()
    {
        // Arrange
        var originalSchema = new Schema(new[] { ("Name", typeof(string)) });
        var metadata = new ColumnMetadata(isNullable: false, description: "Employee age");

        // Act
        var newSchema = originalSchema.WithColumn("Age", typeof(int), metadata);

        // Assert
        var columnMetadata = newSchema.GetColumnMetadata("Age");
        Assert.That(columnMetadata.IsNullable, Is.False);
        Assert.That(columnMetadata.Description, Is.EqualTo("Employee age"));
    }

    [Test]
    public void WithoutColumn_WithExistingColumn_ReturnsNewSchemaWithoutColumn()
    {
        // Arrange
        var originalSchema = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act
        var newSchema = originalSchema.WithoutColumn("Age");

        // Assert
        Assert.That(newSchema.ColumnNames.Count, Is.EqualTo(1));
        Assert.That(newSchema.HasColumn("Name"), Is.True);
        Assert.That(newSchema.HasColumn("Age"), Is.False);
        
        // Original schema should be unchanged
        Assert.That(originalSchema.ColumnNames.Count, Is.EqualTo(2));
        Assert.That(originalSchema.HasColumn("Age"), Is.True);
    }

    [Test]
    public void WithoutColumn_WithNonExistingColumn_ThrowsColumnNotFoundException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => schema.WithoutColumn("Age"));
        Assert.That(ex.ColumnName, Is.EqualTo("Age"));
    }

    [Test]
    public void SelectColumns_WithValidColumns_ReturnsNewSchemaWithSelectedColumns()
    {
        // Arrange
        var originalSchema = new Schema(new[] 
        { 
            ("Name", typeof(string)), 
            ("Age", typeof(int)), 
            ("Salary", typeof(double)) 
        });

        // Act
        var newSchema = originalSchema.SelectColumns(new[] { "Name", "Salary" });

        // Assert
        Assert.That(newSchema.ColumnNames.Count, Is.EqualTo(2));
        Assert.That(newSchema.HasColumn("Name"), Is.True);
        Assert.That(newSchema.HasColumn("Salary"), Is.True);
        Assert.That(newSchema.HasColumn("Age"), Is.False);
        
        // Check order is preserved
        Assert.That(newSchema.ColumnNames[0], Is.EqualTo("Name"));
        Assert.That(newSchema.ColumnNames[1], Is.EqualTo("Salary"));
    }

    [Test]
    public void SelectColumns_WithNonExistingColumn_ThrowsColumnNotFoundException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        var ex = Assert.Throws<ColumnNotFoundException>(() => schema.SelectColumns(new[] { "Age" }));
        Assert.That(ex.ColumnName, Is.EqualTo("Age"));
    }

    [Test]
    public void SelectColumns_WithNullColumnNames_ThrowsArgumentNullException()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => schema.SelectColumns(null!));
    }

    [Test]
    public void IsCompatibleWith_WithIdenticalSchemas_ReturnsTrue()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });
        var schema2 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema1.IsCompatibleWith(schema2), Is.True);
        Assert.That(schema2.IsCompatibleWith(schema1), Is.True);
    }

    [Test]
    public void IsCompatibleWith_WithDifferentColumnCount_ReturnsFalse()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)) });
        var schema2 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema1.IsCompatibleWith(schema2), Is.False);
        Assert.That(schema2.IsCompatibleWith(schema1), Is.False);
    }

    [Test]
    public void IsCompatibleWith_WithDifferentColumnNames_ReturnsFalse()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)) });
        var schema2 = new Schema(new[] { ("Title", typeof(string)) });

        // Act & Assert
        Assert.That(schema1.IsCompatibleWith(schema2), Is.False);
    }

    [Test]
    public void IsCompatibleWith_WithDifferentColumnTypes_ExactMatch_ReturnsFalse()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Age", typeof(int)) });
        var schema2 = new Schema(new[] { ("Age", typeof(long)) });

        // Act & Assert
        Assert.That(schema1.IsCompatibleWith(schema2, requireExactMatch: true), Is.False);
    }

    [Test]
    public void IsCompatibleWith_WithCompatibleNumericTypes_NonExactMatch_ReturnsTrue()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Age", typeof(int)) });
        var schema2 = new Schema(new[] { ("Age", typeof(long)) });

        // Act & Assert
        Assert.That(schema1.IsCompatibleWith(schema2, requireExactMatch: false), Is.True);
    }

    [Test]
    public void IsCompatibleWith_WithNullSchema_ReturnsFalse()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)) });

        // Act & Assert
        Assert.That(schema.IsCompatibleWith(null!), Is.False);
    }

    [Test]
    public void ToString_ReturnsFormattedSchemaDescription()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act
        var result = schema.ToString();

        // Assert
        Assert.That(result, Contains.Substring("Schema"));
        Assert.That(result, Contains.Substring("Name: String"));
        Assert.That(result, Contains.Substring("Age: Int32"));
    }

    [Test]
    public void Equals_WithIdenticalSchemas_ReturnsTrue()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });
        var schema2 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema1.Equals(schema2), Is.True);
        Assert.That(schema1.Equals((object)schema2), Is.True);
    }

    [Test]
    public void Equals_WithDifferentSchemas_ReturnsFalse()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)) });
        var schema2 = new Schema(new[] { ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema1.Equals(schema2), Is.False);
        Assert.That(schema1.Equals((object)schema2), Is.False);
    }

    [Test]
    public void GetHashCode_WithIdenticalSchemas_ReturnsSameHashCode()
    {
        // Arrange
        var schema1 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });
        var schema2 = new Schema(new[] { ("Name", typeof(string)), ("Age", typeof(int)) });

        // Act & Assert
        Assert.That(schema1.GetHashCode(), Is.EqualTo(schema2.GetHashCode()));
    }

    [Test]
    public void CaseInsensitiveColumnAccess_WorksCorrectly()
    {
        // Arrange
        var schema = new Schema(new[] { ("Name", typeof(string)), ("AGE", typeof(int)) });

        // Act & Assert
        Assert.That(schema.HasColumn("name"), Is.True);
        Assert.That(schema.HasColumn("NAME"), Is.True);
        Assert.That(schema.HasColumn("age"), Is.True);
        Assert.That(schema.HasColumn("Age"), Is.True);
        
        Assert.That(schema.GetColumnType("name"), Is.EqualTo(typeof(string)));
        Assert.That(schema.GetColumnType("AGE"), Is.EqualTo(typeof(int)));
        Assert.That(schema.GetColumnType("age"), Is.EqualTo(typeof(int)));
    }
}

[TestFixture]
public class ColumnMetadataTests
{
    [Test]
    public void DefaultConstructor_CreatesMetadataWithDefaults()
    {
        // Act
        var metadata = new ColumnMetadata();

        // Assert
        Assert.That(metadata.IsNullable, Is.True);
        Assert.That(metadata.DefaultValue, Is.Null);
        Assert.That(metadata.Description, Is.Null);
        Assert.That(metadata.Properties, Is.Not.Null);
        Assert.That(metadata.Properties.Count, Is.EqualTo(0));
    }

    [Test]
    public void ParameterizedConstructor_CreatesMetadataWithSpecifiedValues()
    {
        // Arrange
        var properties = new Dictionary<string, object> { { "Format", "yyyy-MM-dd" } };

        // Act
        var metadata = new ColumnMetadata(
            isNullable: false,
            defaultValue: 42,
            description: "Employee ID",
            properties: properties);

        // Assert
        Assert.That(metadata.IsNullable, Is.False);
        Assert.That(metadata.DefaultValue, Is.EqualTo(42));
        Assert.That(metadata.Description, Is.EqualTo("Employee ID"));
        Assert.That(metadata.Properties["Format"], Is.EqualTo("yyyy-MM-dd"));
    }

    [Test]
    public void With_UpdatesSpecifiedProperties()
    {
        // Arrange
        var original = new ColumnMetadata(isNullable: true, description: "Original");

        // Act
        var updated = original.With(isNullable: false, defaultValue: "New Default");

        // Assert
        Assert.That(updated.IsNullable, Is.False);
        Assert.That(updated.DefaultValue, Is.EqualTo("New Default"));
        Assert.That(updated.Description, Is.EqualTo("Original")); // Unchanged
        
        // Original should be unchanged
        Assert.That(original.IsNullable, Is.True);
        Assert.That(original.DefaultValue, Is.Null);
    }

    [Test]
    public void With_WithNullParameters_KeepsOriginalValues()
    {
        // Arrange
        var original = new ColumnMetadata(isNullable: false, defaultValue: 42, description: "Test");

        // Act
        var updated = original.With();

        // Assert
        Assert.That(updated.IsNullable, Is.EqualTo(original.IsNullable));
        Assert.That(updated.DefaultValue, Is.EqualTo(original.DefaultValue));
        Assert.That(updated.Description, Is.EqualTo(original.Description));
    }

    [Test]
    public void ToString_WithDefaultMetadata_ReturnsEmptyBraces()
    {
        // Arrange
        var metadata = new ColumnMetadata();

        // Act
        var result = metadata.ToString();

        // Assert
        Assert.That(result, Is.EqualTo("{}"));
    }

    [Test]
    public void ToString_WithNonNullableMetadata_IncludesNotNull()
    {
        // Arrange
        var metadata = new ColumnMetadata(isNullable: false);

        // Act
        var result = metadata.ToString();

        // Assert
        Assert.That(result, Contains.Substring("NOT NULL"));
    }

    [Test]
    public void ToString_WithDefaultValue_IncludesDefault()
    {
        // Arrange
        var metadata = new ColumnMetadata(defaultValue: 42);

        // Act
        var result = metadata.ToString();

        // Assert
        Assert.That(result, Contains.Substring("DEFAULT 42"));
    }

    [Test]
    public void ToString_WithDescription_IncludesDescription()
    {
        // Arrange
        var metadata = new ColumnMetadata(description: "Employee ID");

        // Act
        var result = metadata.ToString();

        // Assert
        Assert.That(result, Contains.Substring("DESC 'Employee ID'"));
    }

    [Test]
    public void ToString_WithAllProperties_IncludesAllParts()
    {
        // Arrange
        var metadata = new ColumnMetadata(
            isNullable: false,
            defaultValue: 0,
            description: "Counter");

        // Act
        var result = metadata.ToString();

        // Assert
        Assert.That(result, Contains.Substring("NOT NULL"));
        Assert.That(result, Contains.Substring("DEFAULT 0"));
        Assert.That(result, Contains.Substring("DESC 'Counter'"));
    }
}